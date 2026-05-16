import argparse
import os
import torch
import torch.nn.functional as F
try:
    from safetensors.torch import load_file, save_file
except ImportError:
    print("Please install the safetensors library: pip install safetensors")
    exit(1)

def resize_pos_embed(input_path, output_path, old_size, new_size, patch_size, vae_downsample):
    print(f"Loading model: {input_path}")
    state_dict = load_file(input_path)
    
    pos_embed_key = 'patch_embed.pos_embed'
    if pos_embed_key not in state_dict:
        print(f"Error: {pos_embed_key} not found in checkpoint.")
        return
        
    old_pos_embed = state_dict[pos_embed_key] # (1, num_patches, hidden_size)
    hidden_size = old_pos_embed.shape[-1]
    
    old_grid = old_size // vae_downsample // patch_size
    new_grid = new_size // vae_downsample // patch_size
    
    expected_old_len = old_grid * old_grid
    if old_pos_embed.shape[1] != expected_old_len:
        print(f"Warning: Expected old sequence length is {expected_old_len}, but found {old_pos_embed.shape[1]}. Check your parameters.")
    
    print(f"Resizing positional embeddings: {old_grid}x{old_grid} -> {new_grid}x{new_grid}")
    
    # Convert to 2D grid format (B, C, H, W)
    pos_embed_reshaped = old_pos_embed.reshape(1, old_grid, old_grid, hidden_size).permute(0, 3, 1, 2)
    
    # Resize with Bicubic Interpolation
    new_pos_embed = F.interpolate(
        pos_embed_reshaped.float(), # Temporary float() for stable calculations
        size=(new_grid, new_grid),
        mode='bicubic',
        align_corners=False
    ).to(old_pos_embed.dtype)
    
    # Convert back to 1D sequence format
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, hidden_size)
    
    base_patch_len = new_grid * new_grid
    max_patch_len = int(base_patch_len * 1.15)
    
    if new_pos_embed.shape[1] < max_patch_len:
        padded_pos_embed = torch.zeros((1, max_patch_len, hidden_size), dtype=new_pos_embed.dtype)
        padded_pos_embed[:, :new_pos_embed.shape[1], :] = new_pos_embed
        new_pos_embed = padded_pos_embed
    elif new_pos_embed.shape[1] > max_patch_len:
        print(f"Warning: New resolution ({new_pos_embed.shape[1]}) exceeds limits. Cropping...")
        new_pos_embed = new_pos_embed[:, :max_patch_len, :]
    
    # Replace old weight with the new one
    state_dict[pos_embed_key] = new_pos_embed
    
    print(f"Saving new resolution weights: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(state_dict, output_path)
    print("Process completed! You can now use this weight file with the new image_size setting in config.py to resume training.")
    print("WARNING: Do not load the optimizer.bin file when resuming (it will give an error), just load the weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MM-DiT Positional Embedding Resizing Tool")
    parser.add_argument("--input_file", type=str, required=True, help="Path to original model.safetensors weight file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output model.safetensors")
    parser.add_argument("--old_size", type=int, default=256, help="Old resolution (default: 256)")
    parser.add_argument("--new_size", type=int, default=512, help="New resolution (default: 512)")
    parser.add_argument("--patch_size", type=int, default=2, help="MM-DiT Patch Size (default: 2)")
    parser.add_argument("--vae_downsample", type=int, default=8, help="VAE Downsample Factor (default: 8)")
    
    args = parser.parse_args()
    
    resize_pos_embed(args.input_file, args.output_file, args.old_size, args.new_size, args.patch_size, args.vae_downsample)
