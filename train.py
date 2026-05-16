import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable TensorFlow C++ logs
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Disable TensorFlow Python logs
warnings.filterwarnings("ignore") # Ignore all annoying library warnings (librosa, keras, etc.)

import gc
import glob
import json
import math
import shutil
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import SimpleFolderDataset, T2IDataset
from ar_bucketing import ARBucketDataset, ARBatchSampler
from mm_dit import MMDiT, T5Embedder
from vae import VAE

# ---------------------------------------------------------
# Helper: Load VAE Model
# ---------------------------------------------------------
def load_vae(vae_path, device):
    """
    Loads the custom 8-channel VAE using the architecture defined in vae.py.
    """
    model_architecture_config = {
        'base_channels': 128,
        'channel_multipliers': [1, 2, 4, 4],
        'num_residual_blocks_per_level': [2, 2, 2, 4],
        'z_channels': 8
    }
    encoder_params = {**model_architecture_config, 'in_channels': 3}
    decoder_params = {**model_architecture_config, 'out_channels': 3}
    
    vae = VAE(encoder_config=encoder_params, decoder_config=decoder_params)
    
    if os.path.exists(vae_path):
        print(f"Loading VAE weights: {vae_path}")
        checkpoint = torch.load(vae_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            vae.load_state_dict(checkpoint, strict=True)
    else:
        print(f"WARNING: VAE weight file ({vae_path}) not found. Continuing with random weights.")
        
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    return vae

# ---------------------------------------------------------
# Sampling / Inference Function
# ---------------------------------------------------------
@torch.no_grad()
def sample_and_save_grid(model, vae, t5_embedder, config, device, step_idx):
    prompts = getattr(config, "sampling_prompts", [])
    if not prompts:
        return
        
    print(f"\n=> Checkpoint {step_idx} taken, Sampling in progress...")
    model.eval()
    
    def get_hw_from_ar(base_size, aspect_ratio_str):
        if ":" in aspect_ratio_str:
            try:
                w_ratio, h_ratio = map(float, aspect_ratio_str.split(":"))
                ar = w_ratio / h_ratio
            except (ValueError, TypeError):
                ar = 1.0
        else:
            ar = 1.0
        
        area = base_size * base_size
        h = math.sqrt(area / ar)
        w = ar * h
        
        h = int(round(h / 16.0)) * 16
        w = int(round(w / 16.0)) * 16
        return h, w

    generated_images = []
    
    for p in prompts:
        prompt_text = p[0] if isinstance(p, tuple) else p
        ar_str = p[1] if isinstance(p, tuple) else "1:1"
        
        # 1. Text Representation Extraction (Cond & Uncond)
        c_cond, y_cond, _ = t5_embedder([prompt_text], device)
        c_uncond, y_uncond, _ = t5_embedder([""], device)
        
        # 2. CFG Parameters and Output Resizing
        h, w = get_hw_from_ar(config.image_size, ar_str)
        latent_h = h // config.vae_downsample
        latent_w = w // config.vae_downsample
        
        # B, C, H, W
        shape = (1, config.vae_z_channels, latent_h, latent_w)
        x = torch.randn(shape, device=device, dtype=torch.bfloat16)
        
        # 3. Euler Integration (Flow Matching)
        N = config.sampling_steps
        shift = config.timestep_shift_factor
        timesteps = torch.linspace(1.0, 0.0, N + 1)
        timesteps = (shift * timesteps) / (1 + (shift - 1) * timesteps)
        
        for i in range(N):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            t = torch.tensor([t_curr], device=device, dtype=torch.bfloat16)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v_cond = model(x, t, c_cond, y_cond, config)
                v_uncond = model(x, t, c_uncond, y_uncond, config)
                
                # CFG Router
                v_cfg = v_uncond + config.cfg_scale * (v_cond - v_uncond)
            
            # Euler Step
            x = x + v_cfg * dt
            
        # 4. Latent Inference and Decoding
        x = x / config.vae_scaling_factor
        x = x.to(next(vae.parameters()).dtype)
        decoded = vae.decoder(x) # (1, 3, H, W)
        
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        generated_images.append(decoded[0].float().cpu())


    
    # Pad images to the same size (maximum w and h) to make them symmetrical
    max_h = max(img.shape[1] for img in generated_images)
    max_w = max(img.shape[2] for img in generated_images)
    
    padded_images = []
    for img in generated_images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        padded_img = F.pad(img, padding, value=0.0)
        padded_images.append(padded_img)
        
    stacked_images = torch.stack(padded_images)
    
    os.makedirs(os.path.join(config.output_path, "samples"), exist_ok=True)
    grid = vutils.make_grid(stacked_images, nrow=4, padding=4)
    path = os.path.join(config.output_path, "samples", f"sample_step_{step_idx}.png")
    vutils.save_image(grid, path)
    
    print(f"=> Sampling completed: {path}")
    
    # VRAM Cleanup
    try:
        del x, c_cond, y_cond, c_uncond, y_uncond, decoded, grid
    except NameError:
        pass
        
    gc.collect()
    torch.cuda.empty_cache()
    
    model.train()

# ---------------------------------------------------------
# Main Training Loop (Flow Matching)
# ---------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device in use: {device}")
    
    # 1. Load Settings
    config = Config()
    os.makedirs(config.output_path, exist_ok=True)
    
    # 2. Load Models
    print("Preparing models...")
    vae = load_vae(config.vae_path, device)
    
    # T5
    print(f"Loading T5: {config.t5_path}")
    t5_embedder = T5Embedder(config.t5_path, config.hidden_size).to(device)
    
    # MM-DiT Model
    print("Creating MM-DiT Architecture...")
    model = MMDiT(config).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=> Total Trainable Parameter Count (MM-DiT): {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print("Reading Data...")
    
    # Collect dataset configurations
    dataset_configs = getattr(config, 'datasets', [])
    
    if getattr(config, 'use_ar_bucketing', False):
        print("=> Aspect Ratio Bucketing is ACTIVE! Using ARBucketDataset & ARBatchSampler.")
        if not dataset_configs:
            print("Training stopped because there is no dataset configuration.")
            return
            
        bucket_json_path = os.path.join(config.output_path, "ar_buckets.json")
        train_dataset = ARBucketDataset(dataset_configs, bucket_json=bucket_json_path, caption_dropout=config.caption_dropout)
    else:
        datasets_to_concat = []
        
        for ds_config in dataset_configs:
            if "images_dir" in ds_config:
                datasets_to_concat.append(SimpleFolderDataset(ds_config, image_size=config.image_size, caption_dropout=config.caption_dropout))
            elif "tar_dir" in ds_config:
                datasets_to_concat.append(T2IDataset(ds_config, image_size=config.image_size, caption_dropout=config.caption_dropout))
            
        if len(datasets_to_concat) == 0:
             print("Training stopped because there are no images/folders in the dataset.")
             return
        elif len(datasets_to_concat) == 1:
            train_dataset = datasets_to_concat[0]
        else:
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset(datasets_to_concat)
            print(f"=> A total of {len(train_dataset)} images merged (ConcatDataset).")
    
    if len(train_dataset) == 0:
         print("Training stopped because there are no images in the dataset.")
         return
         
    # --- RESUME LOGIC & JSON STATE READING ---
    start_epoch = 0
    global_optim_step = 0
    total_images_seen = 0
    items_to_skip_this_epoch = 0
    state_json_path = os.path.join(config.output_path, "training_state.json")
    
    if getattr(args, 'resume_from_checkpoint', None):
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            all_checkpoints = glob.glob(os.path.join(config.output_path, "checkpoints", "mm_dit_step_*.pt")) + glob.glob(os.path.join(config.output_path, "mm_dit_step_*.pt"))
            all_checkpoints_dirs = [d for d in glob.glob(os.path.join(config.output_path, "checkpoints", "checkpoint_step_*")) if os.path.isdir(d)] + [d for d in glob.glob(os.path.join(config.output_path, "checkpoint_step_*")) if os.path.isdir(d)]
            
            latest_step = -1
            latest_path = None
            
            for ckpt in all_checkpoints:
                try:
                    step = int(os.path.basename(ckpt).split('_step_')[1].split('.pt')[0])
                    if step > latest_step:
                        latest_step = step
                        latest_path = ckpt
                except (ValueError, IndexError): pass
            for ckpt in all_checkpoints_dirs:
                try:
                    step = int(os.path.basename(ckpt).split('_step_')[1])
                    if step > latest_step:
                        latest_step = step
                        latest_path = ckpt
                except: pass
                
            if latest_path:
                resume_path = latest_path
            else:
                resume_path = None
                print("Latest checkpoint not found, starting from scratch.")
                
        if resume_path and os.path.exists(resume_path):
            print(f"=> Loading Checkpoint: {resume_path}")
            if os.path.isdir(resume_path):
                try:
                    from safetensors.torch import load_file
                    model.load_state_dict(load_file(os.path.join(resume_path, "model.safetensors")))
                except ImportError:
                    print("WARNING: safetensors not installed, falling back to torch.load!")
                    model.load_state_dict(torch.load(os.path.join(resume_path, "model.safetensors"), map_location='cpu', weights_only=False))
                
                optimizer.load_state_dict(torch.load(os.path.join(resume_path, "optimizer.bin"), map_location='cpu', weights_only=False))
                
                with open(os.path.join(resume_path, "training_state.json"), "r") as f:
                    ckpt_state = json.load(f)
                global_optim_step = ckpt_state.get("step", 0)
                start_epoch_ckpt = ckpt_state.get("epoch", 1) - 1
            else:
                try:
                    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    global_optim_step = checkpoint['step']
                    start_epoch_ckpt = checkpoint.get('epoch', 1) - 1
                except Exception as e:
                    print(f"WARNING: Could not be read as a standard PyTorch checkpoint: {e}")
                    print("Assuming this might be a Safetensors file and retrying...")
                    from safetensors.torch import load_file
                    model.load_state_dict(load_file(resume_path))
                    print("SUCCESS: Model weights loaded from Safetensors file (excluding Optimizer/step history).")
                    global_optim_step = 0
                    start_epoch_ckpt = 0
            
            # Extracting images_seen from JSON (To protect against Batch/accumulation changes)
            if os.path.exists(state_json_path):
                with open(state_json_path, 'r') as f:
                    state_data = json.load(f)
                    start_epoch = state_data.get("epoch", start_epoch_ckpt)
                    items_to_skip_this_epoch = state_data.get("images_seen_this_epoch", 0)
                    total_images_seen = state_data.get("total_images_seen", 0)
                    print(f"=> JSON State Loaded! Total Images Seen: {total_images_seen}, Continuing from Epoch: {start_epoch+1}")
            else:
                start_epoch = start_epoch_ckpt
                
    # --- Force Learning Rate (Mandatory LR Assignment) ---
    # If a new LR is provided via terminal (to override the optimizer's current LR during resume)
    if getattr(args, 'force_lr', None) is not None:
        print(f"\n=> WARNING: Learning Rate value is being forced to {args.force_lr} via '--force-lr' argument!")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.force_lr
                
    # Tensorboard Logging
    writer = SummaryWriter(os.path.join(config.output_path, "logs"))
    
    # 4. Training Loop
    local_step = 0 # Target batch indices (for Gradient Accumulation control)
    print("Training Begins!")
    optimizer.zero_grad() # Reset at the start of training
    
    accumulated_loss = 0.0 # To accumulate the loss of all micro-batches
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        g = torch.Generator()
        g.manual_seed(2147483647 + epoch) 
        if getattr(config, 'use_ar_bucketing', False):
            batch_sampler = ARBatchSampler(train_dataset, batch_size=config.batch_size, drop_last=True)
            train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=6, generator=g)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=6, generator=g)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Tracker for how many images were examined in the current epoch
        images_seen_this_epoch_tracker = 0
        if items_to_skip_this_epoch > 0:
            images_seen_this_epoch_tracker = items_to_skip_this_epoch
            
        for batch_idx, (batch_images, batch_captions) in enumerate(progress_bar):
            B = batch_images.size(0)
            
            # Resume (skipping) algorithm from where it left off
            if items_to_skip_this_epoch > 0:
                if items_to_skip_this_epoch >= B:
                    items_to_skip_this_epoch -= B
                    local_step += 1 # For gradient cycle synchronization in resume scenario
                    continue
                else:
                    # Partial shift (If batch was changed)
                    items_to_skip_this_epoch = 0 
            
            images_seen_this_epoch_tracker += B
            total_images_seen += B
            local_step += 1
            batch_images = batch_images.to(device)
            B = batch_images.size(0)
            
            # --- 1. Obtaining Latent with VAE (x_0: Data) ---
            with torch.no_grad():
                mu, logvar = vae.encoder(batch_images)
                # Standard reparameterization
                x_0 = vae.reparameterize(mu, logvar)
                
                # Scaled Latent (With specified scale parameter)
                x_0 = x_0 * config.vae_scaling_factor
                
            # --- 2. T5 Text Representation Extraction ---
            c, pooled_y, _ = t5_embedder(batch_captions, device)
            
            # --- 3. Flow Matching Timestep (t) and (x_1) Noise Generation ---
            shift_log = math.log(config.timestep_shift_factor)
            t = torch.sigmoid(torch.randn((B,), device=device) + shift_log) 
            
            x_1 = torch.randn_like(x_0)
            t_expanded = t.view(B, 1, 1, 1)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            v_target = x_1 - x_0
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                v_pred = model(x_t, t, c, pooled_y, config)
                loss = F.mse_loss(v_pred, v_target)
                loss = loss / config.gradient_accumulation_steps
            
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Optimizer Step (Only when ACCUMULATION_STEPS is reached)
            if local_step % config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                global_optim_step += 1
                
                # --- Tensorboard Logs ---
                if global_optim_step % config.log_steps == 0:
                    writer.add_scalar("Loss/train_velocity", accumulated_loss, global_optim_step)
                    writer.add_scalar("Metrics/Grad_Norm", grad_norm.item(), global_optim_step)
                    writer.add_scalar("Epochs/epoch", epoch + 1, global_optim_step)
                    
                accumulated_loss = 0.0
                    
                # --- Save Core (Steps - Based) ---
                if global_optim_step % config.checkpoint_freq == 0:
                    checkpoint_dir = os.path.join(config.output_path, "checkpoints", f"checkpoint_step_{global_optim_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    try:
                        from safetensors.torch import save_file
                        save_file(model.state_dict(), os.path.join(checkpoint_dir, "model.safetensors"))
                    except ImportError:
                        print("WARNING: safetensors library not found, saving with torch (model.safetensors)")
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.safetensors"))
                        
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.bin"))
                    
                    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
                        json.dump({
                            "step": global_optim_step,
                            "epoch": epoch + 1
                        }, f)
                        
                    shutil.copy("config.py", os.path.join(checkpoint_dir, "config.py"))
                        
                    print(f"\nWeights saved as separate files (in folder): {checkpoint_dir}")
                    
                    # Clean up old checkpoints (max_checkpoints)
                    all_checkpoints = glob.glob(os.path.join(config.output_path, "checkpoints", "mm_dit_step_*.pt")) + glob.glob(os.path.join(config.output_path, "mm_dit_step_*.pt"))
                    all_dirs = [d for d in glob.glob(os.path.join(config.output_path, "checkpoints", "checkpoint_step_*")) if os.path.isdir(d)] + [d for d in glob.glob(os.path.join(config.output_path, "checkpoint_step_*")) if os.path.isdir(d)]
                    
                    all_combined = all_checkpoints + all_dirs
                    if len(all_combined) > config.max_checkpoints:
                        def get_step(path):
                            if path.endswith('.pt'):
                                return int(os.path.basename(path).split('_step_')[1].split('.pt')[0])
                            else:
                                return int(os.path.basename(path).split('_step_')[1])
                                
                        all_combined.sort(key=get_step)
                        oldest = all_combined[0]
                        if os.path.isdir(oldest):
                            shutil.rmtree(oldest)
                        else:
                            os.remove(oldest)
                        
                    # If there are prompts in config, test the model's capability
                    sample_and_save_grid(model, vae, t5_embedder, config, device, global_optim_step)
                    
                    # Reliable JSON State Logging (Resilient to Accumulation/Batch size changes)
                    with open(state_json_path, 'w') as f:
                        json.dump({
                            "epoch": epoch,
                            "images_seen_this_epoch": images_seen_this_epoch_tracker,
                            "total_images_seen": total_images_seen,
                            "global_optim_step": global_optim_step
                        }, f)
            
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{(loss.item()*config.gradient_accumulation_steps):.4f}", "optim_step": global_optim_step, "lr": f"{current_lr:g}"})
            
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help='The .pt weight to resume from or "latest"')
    parser.add_argument('--force-lr', type=float, default=None, help='Force change LR value during training (or when resuming) (e.g., 1e-4)')
    args = parser.parse_args()
    train(args)
