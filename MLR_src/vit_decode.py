import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)

class ViTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, tgt_mask=None):
        # Self attention with causal mask
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class ViTDecoder(nn.Module):
    def __init__(self, latent_dim, num_tokens, embed_dim, num_layers, n_heads, patch_size, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens  # initial number of tokens (could be a small spatial grid)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Map latent vector to an initial sequence of tokens
        self.latent_to_tokens = nn.Linear(latent_dim, num_tokens * embed_dim)
        
        # Positional embeddings for the token sequence (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, embed_dim))
        
        # Transformer decoder layers (stacked)
        self.layers = nn.ModuleList([
            ViTDecoderLayer(embed_dim, n_heads)
            for _ in range(num_layers)
        ])
        
        # Final head to map token representations to patch pixel values
        # For instance, each token maps to a flattened patch of pixels
        patch_dim = patch_size * patch_size * 3  # 3 channels (RGB)
        self.to_patch = nn.Linear(embed_dim, patch_dim)
        
    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        """
        batch_size = z.size(0)
        # Project latent vector into token sequence:
        tokens = self.latent_to_tokens(z)  # (batch_size, num_tokens * embed_dim)
        tokens = tokens.view(batch_size, self.num_tokens, self.embed_dim)  # reshape tokens
        
        # Add positional embeddings:
        tokens = tokens + self.pos_embedding.unsqueeze(0)
        
        # Create causal mask (upper triangular) for autoregressive decoding
        seq_len = tokens.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=z.device) * float('-inf'), diagonal=1)
        
        # Pass through decoder layers:
        # Note: nn.MultiheadAttention expects input shape (seq_len, batch_size, embed_dim)
        x = tokens.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, tgt_mask=tgt_mask)
        x = x.transpose(0, 1)  # (batch_size, num_tokens, embed_dim)
        
        # Map each token to a patch
        patches = self.to_patch(x)  # (batch_size, num_tokens, patch_dim)
        
        # Reshape patches into image grid:
        grid_size = int(self.num_tokens ** 0.5)
        img = patches.view(batch_size, grid_size, grid_size, self.patch_size, self.patch_size, 3)
        img = img.permute(0, 5, 1, 3, 2, 4)  # rearrange axes to (batch, channels, grid_h, patch_h, grid_w, patch_w)
        img = img.contiguous().view(batch_size, 3, grid_size * self.patch_size, grid_size * self.patch_size)
        return img

# test
'''latent_dim = 256
num_tokens = 16  # e.g., a 4x4 grid of tokens
embed_dim = 512
num_layers = 6
n_heads = 8
patch_size = 8
img_size = 32  # final output size

decoder = ViTDecoder(latent_dim, num_tokens, embed_dim, num_layers, n_heads, patch_size, img_size)
z_sample = torch.randn(4, latent_dim)
reconstructed_img = decoder(z_sample)
print(reconstructed_img.shape)  # Expected: (4, 3, 32, 32)'''
