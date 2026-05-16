import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------
# Timestep Embedder (MLP + sinusoidal_encoding + timestep)
# ---------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_emb_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.freq_emb_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

# ---------------------------------------------------------
# Helper Modulation Function
# ---------------------------------------------------------
def modulate(x, shift, scale):
    """
    Modulation function.
    x = x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift

# ---------------------------------------------------------
# RMS Norm (For Attention Key/Query)
# ---------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x / torch.sqrt(norm + self.eps) * self.weight

class MMDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # --- c (text) Path ---
        self.ln_c1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_c = nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q_c = RMSNorm(hidden_size)
        self.norm_k_c = RMSNorm(hidden_size)
        self.proj_c = nn.Linear(hidden_size, hidden_size)
        
        self.ln_c2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_c = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        
        # --- x Path ---
        self.ln_x1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_x = nn.Linear(hidden_size, hidden_size * 3)
        self.norm_q_x = RMSNorm(hidden_size)
        self.norm_k_x = RMSNorm(hidden_size)
        self.proj_x = nn.Linear(hidden_size, hidden_size)
        
        self.ln_x2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        
        # --- y (Global Conditioning) Path (Linear only) ---
        # SiLU + Linear. Modulations for both c and x (6 vectors for c, 6 for x = 12 total)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size)
        )
        
    def forward(self, c, x, y):
        # 1. Extract Modulation Vectors
        mod_params = self.adaLN_modulation(y).chunk(12, dim=-1)
        alpha_c, beta_c, gamma_c, delta_c, eps_c, zeta_c = mod_params[0:6]
        alpha_x, beta_x, gamma_x, delta_x, eps_x, zeta_x = mod_params[6:12]
        
        # Expanding tensor dimension for broadcasting (B, 1, hidden_size)
        alpha_c, beta_c, gamma_c, delta_c, eps_c, zeta_c = [v.unsqueeze(1) for v in (alpha_c, beta_c, gamma_c, delta_c, eps_c, zeta_c)]
        alpha_x, beta_x, gamma_x, delta_x, eps_x, zeta_x = [v.unsqueeze(1) for v in (alpha_x, beta_x, gamma_x, delta_x, eps_x, zeta_x)]
        
        # ----- ATTENTION SECTION -----
        # Modulation and Q,K,V for c
        c_mod = modulate(self.ln_c1(c), shift=beta_c, scale=alpha_c)
        qkv_c = self.qkv_c(c_mod).chunk(3, dim=-1)
        q_c, k_c, v_c = self.norm_q_c(qkv_c[0]), self.norm_k_c(qkv_c[1]), qkv_c[2]
        
        # Modulation and Q,K,V for x
        x_mod = modulate(self.ln_x1(x), shift=beta_x, scale=alpha_x)
        qkv_x = self.qkv_x(x_mod).chunk(3, dim=-1)
        q_x, k_x, v_x = self.norm_q_x(qkv_x[0]), self.norm_k_x(qkv_x[1]), qkv_x[2]
        
        # Joint Attention: concatenate q, k, v for c and x.
        q = torch.cat([q_c, q_x], dim=1) # (B, Seq_c + Seq_x, hidden_size)
        k = torch.cat([k_c, k_x], dim=1)
        v = torch.cat([v_c, v_x], dim=1)
        
        # Multi-head splitting
        B, Seq, _ = q.shape
        q = q.view(B, Seq, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2).contiguous()
        k = k.view(B, Seq, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, Seq, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2).contiguous()
        
        # Flash Attention (PyTorch automatically selects the optimal backend: Flash, Memory-Efficient, or Math)
        attn_out = F.scaled_dot_product_attention(q, k, v)
            
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Seq, self.hidden_size)
        
        # Split outputs back into c and x.
        len_c = c.shape[1]
        out_c, out_x = attn_out[:, :len_c, :], attn_out[:, len_c:, :]
        
        # Linear & Modulation -> Residual Add
        c = c + gamma_c * self.proj_c(out_c)
        x = x + gamma_x * self.proj_x(out_x)
        
        # ----- MLP SECTION -----
        # Modulation and MLP for c
        c_mod2 = modulate(self.ln_c2(c), shift=eps_c, scale=delta_c)
        c = c + zeta_c * self.mlp_c(c_mod2)
        
        # Modulation and MLP for x
        x_mod2 = modulate(self.ln_x2(x), shift=eps_x, scale=delta_x)
        x = x + zeta_x * self.mlp_x(x_mod2)
        
        return c, x

# ---------------------------------------------------------
# Patching and Unpatching Functions
# ---------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, patch_size=2, max_len=4096):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(in_channels * patch_size**2, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))

    def forward(self, x):
        # x: Noised latent -> (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Splitting image size into patches (Patching)
        x = x.view(B, C, H//P, P, W//P, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H//P) * (W//P), C * P * P)
        
        # Adding Linear Projection and Positional Encoding
        x = self.linear(x)
        x = x + self.pos_embed[:, :x.shape[1], :]
        return x

def unpatchify(x, h, w, patch_size, out_channels):
    """
    x: (B, Seq, out_channels * patch_size * patch_size)
    """
    B = x.shape[0]
    P = patch_size
    H_p, W_p = h // P, w // P
    x = x.view(B, H_p, W_p, out_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, out_channels, h, w)
    return x

# ---------------------------------------------------------
# Final Output Block
# ---------------------------------------------------------
class FinalBlock(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        
    def forward(self, x, y):
        # Modulation by y
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        # After the linear operation, it takes a form suitable for unpatching.
        x = self.linear(x)
        return x

# ---------------------------------------------------------
# T5 Wrapper
# ---------------------------------------------------------
class T5Embedder(nn.Module):
    def __init__(self, t5_path, hidden_size):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path, legacy=False)
        self.t5_model = T5EncoderModel.from_pretrained(t5_path)
        
        self.t5_model.eval()
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
        t5_dim = self.t5_model.config.d_model
        
    def forward(self, texts, device):
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        attn_mask = tokens.attention_mask
        
        with torch.no_grad():
            outputs = self.t5_model(input_ids=tokens.input_ids, attention_mask=attn_mask)
            seq_embs = outputs.last_hidden_state # (B, Seq, t5_dim)
            
        # y output (Global representative via Mean Pooling)
        seq_len = attn_mask.sum(dim=1, keepdim=True).to(seq_embs.dtype)
        pooled = (seq_embs * attn_mask.unsqueeze(-1)).sum(dim=1) / seq_len.clamp(min=1e-9)
        
        return seq_embs, pooled, attn_mask

# ---------------------------------------------------------
# MM-DiT Main Architecture
# ---------------------------------------------------------
class MMDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.vae_z_channels = config.vae_z_channels
        
        base_patch_len = (config.image_size // config.vae_downsample // config.patch_size) ** 2
        
        # In case of AR bucketing, the number of patches may increase by 10-15% due to aspect ratios.
        # For example, for 512 resolution, the largest bucket 640x448 generates 1120 patches (instead of 1024).
        # We only add the required 15% tolerance.
        max_patch_len = int(base_patch_len * 1.15) if getattr(config, 'use_ar_bucketing', False) else base_patch_len
        
        # Patching process
        self.patch_embed = PatchEmbed(config.vae_z_channels, config.hidden_size, config.patch_size, max_len=max_patch_len)
        
        # Timestep (Time) Module
        self.time_embed = TimestepEmbedder(config.hidden_size)
        
        # c Path Projection (For T5 Sequence Tensor)
        self.c_proj = nn.Linear(config.t5_dim, config.hidden_size)
        
        # y Path MLP (For T5 Pooled Tensor)
        self.y_mlp = nn.Sequential(
            nn.Linear(config.t5_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.blocks = nn.ModuleList([
             MMDiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
             for _ in range(config.depth)
        ])
        
        # Final Unpatching Layer Series
        self.final_block = FinalBlock(config.hidden_size, config.patch_size, config.vae_z_channels)
        
    def forward(self, x, t, c, pooled_text, config):
        """
        x: Noised latent (B, C, H, W)
        t: Continuous timestep (B,)
        c: Raw sequence T5 embeddings (B, seq_len, t5_dim)
        pooled_text: Pooled T5 embeddings (B, t5_dim) - This will go into MLP.
        config: to access gradient_checkpointing info.
        """
        B, C, H, W = x.shape
        
        # 0. Project c to model's hidden_size
        c = self.c_proj(c)
        
        # 1. Noised Latent -> Patching and Positional Encoding (x output)
        x = self.patch_embed(x)
        
        # 2. Preparation of y Output Vector
        t_emb = self.time_embed(t)
        pooled_emb = self.y_mlp(pooled_text)
        y = pooled_emb + t_emb # Combining the structure passed through two MLPs to create 'y'.
        
        for block in self.blocks:
             if self.training and getattr(config, "gradient_checkpointing", False):
                 c, x = checkpoint(block, c, x, y, use_reentrant=False)
             else:
                 c, x = block(c, x, y)
             
        # 4. Final Layers
        # x => modulation by y => linear
        x_out = self.final_block(x, y)
        
        # 5. Unpatching
        out = unpatchify(x_out, H, W, self.patch_size, self.vae_z_channels)
        return out
