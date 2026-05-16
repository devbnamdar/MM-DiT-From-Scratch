import os

class Config:
    vae_path = "path/to/your/vae_model.pth"
    t5_path = "t5-base"
    datasets = [
        {
            # Use "images_dir": for standard folders or "tar_dir": for WebDataset-style tar archives.
            "tar_dir": r"/path/to/your/dataset_tar_folder",
            "caption_jsonl": r"/path/to/your/captions.jsonl"
        }
    ]
    output_path = "yourOutputFolder"
    
    image_size = 256
    use_ar_bucketing = False # Use Aspect Ratio Bucketing? If yes, ar_buckets.json is required.
    vae_downsample = 8
    latent_size = image_size // vae_downsample
    vae_z_channels = 8
    vae_scaling_factor = 0.99402
    
    patch_size = 2
    hidden_size = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    
    t5_dim = 768
    
    learning_rate = 1e-4
    batch_size = 1
    num_epochs = 1
    caption_dropout = 0.2
    
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    
    # --- Checkpoint and Log Settings ---
    checkpoint_freq = 50 # training step frequency for saving the model
    max_checkpoints = 4 # maximum number of checkpoints to save in same folder
    log_steps = 4 # training step frequency for logging to TensorBoard
    
    timestep_shift_factor = 1.0
    
    cfg_scale = 3.5
    sampling_steps = 25
    sampling_prompts = [
        ("prompt sample1", "1:1"),
        ("prompt sample2", "4:3"),
    ]  # this will be generate image samples during the training every "checkpoint_freq"