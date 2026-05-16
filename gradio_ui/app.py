import os
import sys
import torch
import torchvision.utils as vutils
import gradio as gr
from PIL import Image
import datetime

# Add project root to sys.path so we can import modules like mm_dit etc.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from mm_dit import MMDiT, T5Embedder
from vae import VAE

def load_vae_for_inference(vae_path, device):
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
        print(f"[*] Loading VAE weights: {vae_path}")
        checkpoint = torch.load(vae_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            vae.load_state_dict(checkpoint, strict=True)
    else:
        print(f"[!] WARNING: VAE weight file ({vae_path}) not found. Continuing with random weights.")
        
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    return vae

# ==========================================
# Global State Management for Models
# ==========================================
class ModelState:
    vae = None
    t5_embedder = None
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = None
    
    # To track model changes
    last_vae_path = None
    last_t5_path = None

state = ModelState()

# Architecture configuration
class AppConfig:
    vae_path = "vae_models/Nova_ae_f8.pth"
    t5_path = "t5-base"
    
    image_size = 512
    vae_downsample = 8
    vae_z_channels = 8
    vae_scaling_factor = 0.99402
    
    timestep_shift_factor = 3.0
    
    patch_size = 2
    hidden_size = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    t5_dim = 768

state.config = AppConfig()

def load_models(base_model_path, vae_path):
    try:
        t5_path = state.config.t5_path
        print("Loading models...")
        
        # Convert to absolute path if paths are given relative to project root
        if vae_path and not os.path.isabs(vae_path):
            abs_vae_path = os.path.join(project_root, vae_path)
        else:
            abs_vae_path = vae_path

        # 1. Load VAE
        if state.vae is None or state.last_vae_path != abs_vae_path:
            print(f"[*] Loading VAE: {abs_vae_path}")
            state.vae = load_vae_for_inference(abs_vae_path, state.device)
            state.last_vae_path = abs_vae_path
            
        # 2. Load T5
        if state.t5_embedder is None or state.last_t5_path != t5_path:
            print(f"[*] Loading T5 Embedder: {t5_path}")
            state.t5_embedder = T5Embedder(t5_path, state.config.hidden_size).to(state.device)
            state.last_t5_path = t5_path

        # 3. Loading Base Model (MM-DiT)
        print(f"[*] Loading Base Model: {base_model_path}")
        state.model = MMDiT(state.config).to(state.device)
        
        if not os.path.exists(base_model_path):
            return f"Error: Base model file not found: {base_model_path}"
            
        try:
            from safetensors.torch import load_file
            state.model.load_state_dict(load_file(base_model_path), strict=False)
        except ImportError:
            state.model.load_state_dict(torch.load(base_model_path, map_location='cpu', weights_only=False), strict=False)

        return "All models loaded successfully! You can now switch to the 'Generation' tab."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}"

import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def generate(prompt, negative_prompt, cfg_scale, steps, seed, width, height, batch_size, batch_count, shift_factor):
    if state.model is None or state.vae is None or state.t5_embedder is None:
        raise gr.Error("Please load the models from the 'Settings' tab first!")
        
    state.model.eval()
    
    # Text Embedding (calculated only once)
    c_cond, y_cond, _ = state.t5_embedder([prompt] * int(batch_size), state.device)
    c_uncond, y_uncond, _ = state.t5_embedder([negative_prompt] * int(batch_size), state.device)
    
    latent_width = int(width) // state.config.vae_downsample
    latent_height = int(height) // state.config.vae_downsample
    shape = (int(batch_size), state.config.vae_z_channels, latent_height, latent_width)
    
    N = int(steps)
    shift = float(shift_factor)
    timesteps = torch.linspace(1.0, 0.0, N + 1)
    timesteps = (shift * timesteps) / (1 + (shift - 1) * timesteps)
    
    # Create folder and save (A1111 style daily folders)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(project_root, "generated_samples", today_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Finding sequence number in A1111 style
    max_seq = -1
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            parts = filename.split("-")
            if len(parts) >= 2 and parts[0].isdigit():
                seq = int(parts[0])
                if seq > max_seq:
                    max_seq = seq
    next_seq = max_seq + 1
    
    image_paths = []
    base_seed = int(seed) if seed and str(seed).strip() != "" else torch.seed()
    
    for bc in range(int(batch_count)):
        current_seed = base_seed + bc
        torch.manual_seed(current_seed)
        if state.device.type == 'cuda':
            torch.cuda.manual_seed_all(current_seed)
        print(f"--- Batch Count {bc+1}/{int(batch_count)} | Seed: {current_seed} ---")
        
        x = torch.randn(shape, device=state.device, dtype=torch.bfloat16)
        
        # Inference Loop (Pass 1 - Base Generation)
        for i in range(N):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            t_tensor = torch.tensor([t_curr] * int(batch_size), device=state.device, dtype=torch.bfloat16)
            
            with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16):
                v_cond = state.model(x, t_tensor, c_cond, y_cond, state.config)
                v_uncond = state.model(x, t_tensor, c_uncond, y_uncond, state.config)
                # Classifier-Free Guidance
                v_cfg = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
            
            # Euler update
            x = x + v_cfg * dt
            

            
        # Decode
        print("=> Decoding VAE...")
        x_decoded = x / state.config.vae_scaling_factor
        x_decoded = x_decoded.to(next(state.vae.parameters()).dtype)
        decoded = state.vae.decoder(x_decoded)
        decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
        
        for b in range(int(batch_size)):
            item_seed = current_seed + b
            output_name = f"{next_seq:05d}-{item_seed}.png"
            output_path = os.path.join(output_dir, output_name)
            vutils.save_image(decoded[b].float(), output_path, padding=0)
            image_paths.append(output_path)
            next_seq += 1
            
    print(f"[*] Generation complete. Total of {len(image_paths)} images created.")
    return image_paths

# ==========================================
# Gradio Interface
# ==========================================
custom_css = '''
.gradio-container { font-family: 'Inter', sans-serif; }
#gallery { min-height: 500px; }
'''

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=custom_css, title="MMDiT Image Generator Studio") as app:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>T2i Studio</h1>")
    
    with gr.Tabs():
        with gr.Tab("🖼️ Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image you want to generate in detail...")
                    negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Elements you don't want in the image (e.g., blurry, worst quality, text)...")
                    
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Width")
                        height = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Height")
                    
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Sampling Steps")
                        cfg_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=3.5, label="CFG Scale")
                        shift_factor = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Timestep Shift Factor")
                    
                    with gr.Row():
                        batch_count = gr.Slider(minimum=1, maximum=50, step=1, value=1, label="Batch Count")
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Batch Size")
                        seed = gr.Textbox(label="Seed", placeholder="Leave empty for random")
                        

                    generate_btn = gr.Button("🚀 Generate Image(s)", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery")
                    
        with gr.Tab("⚙️ Settings (Model Paths)"):
            gr.Markdown("### 📂 Model Paths and Settings")
            gr.Markdown("Fill in the model paths below completely. You cannot generate images without completing the model loading.")
            
            with gr.Group():
                gr.Markdown("#### Base Model")
                base_model_path = gr.Textbox(label="Base Model Path (.safetensors or .pth)", value="", placeholder="Enter the path to your model.safetensors checkpoint file")
            with gr.Group():
                gr.Markdown("#### Side Components (VAE)")
                vae_path = gr.Textbox(label="VAE Path (Relative to project directory or absolute path)", value="vae_models/Nova_ae_f8.pth")
                
            load_btn = gr.Button("💾 Load Models to GPU", variant="secondary")
            load_status = gr.Textbox(label="Loading Status", interactive=False)
            
    # Event Bindings
    load_btn.click(
        fn=load_models,
        inputs=[base_model_path, vae_path],
        outputs=load_status,
        api_name=False
    )
    
    generate_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, cfg_scale, steps, seed, width, height, batch_size, batch_count, shift_factor],
        outputs=gallery,
        api_name=False,
        show_progress="always"
    )

if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
