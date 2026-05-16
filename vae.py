import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(1, 2)
        x = self.attention(x)
        x = x.transpose(1, 2).reshape((n, c, h, w))
        x += residue
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)

class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, channel_multipliers=[1, 2, 4, 4],
                 num_residual_blocks_per_level=[2, 2, 2, 2], z_channels=4):
        super().__init__()
        self.channel_multipliers = channel_multipliers
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        layers = []
        current_channels = base_channels
        for i, multiplier in enumerate(self.channel_multipliers):
            out_channels = base_channels * multiplier
            for _ in range(num_residual_blocks_per_level[i]):
                layers.append(VAE_ResidualBlock(current_channels, out_channels))
                current_channels = out_channels
            if i < len(self.channel_multipliers) - 1:
                layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))
        layers.extend([
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_AttentionBlock(current_channels),
            VAE_ResidualBlock(current_channels, current_channels),
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, 2 * z_channels, kernel_size=3, padding=1)
        ])
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv_in(x)
        for module in self.model:
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        return mean, log_variance

class VAE_Decoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=128, channel_multipliers=[1, 2, 4, 4],
                 num_residual_blocks_per_level=[2, 2, 2, 2], z_channels=4):
        super().__init__()
        current_channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(z_channels, current_channels, kernel_size=3, padding=1)
        layers = [
            VAE_ResidualBlock(current_channels, current_channels),
            VAE_AttentionBlock(current_channels),
            VAE_ResidualBlock(current_channels, current_channels)
        ]
        for i in reversed(range(len(channel_multipliers))):
            out_channels_level = base_channels * channel_multipliers[i]
            for _ in range(num_residual_blocks_per_level[i] + 1):
                layers.append(VAE_ResidualBlock(current_channels, out_channels_level))
                current_channels = out_channels_level
            if i > 0:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='bicubic'),
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1)
                ])
        
        layers.extend([
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        ])
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv_in(x)
        for module in self.model:
            x = module(x)
        return x

class VAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = VAE_Encoder(**encoder_config)
        self.decoder = VAE_Decoder(**decoder_config)
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar, z
