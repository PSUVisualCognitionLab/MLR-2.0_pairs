# ViT implementation written by lucidrains, modified by ian for pytorch 1.3.1 : https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PatchEmbed(nn.Module):
    def __init__(
        self, *,
        image_size,
        patch_size,
        dim,
        channels = 3,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        # Compute grid dimensions and total patches
        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        num_patches = self.grid_h * self.grid_w
        patch_dim = channels * patch_height * patch_width  # e.g. 3*7*7 = 147

        # Patch embedding: use einops Rearrange to split image into patches
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return self.to_patch_embedding(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(
        self, *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        patch_embed,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        masked: str = None    # New: 'None' (default), 'center', or 'outer'
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        # Compute grid dimensions and total patches
        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width
        num_patches = self.grid_h * self.grid_w
        patch_dim = channels * patch_height * patch_width  # e.g. 3*7*7 = 147

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # Patch embedding: use einops Rearrange to split image into patches
        self.to_patch_embedding = patch_embed

        # Positional embeddings: for all patches plus the cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

        # Save masking mode and precompute patch coordinates if needed.
        self.masked = masked  # one of: None, 'center', 'outer'
        if self.masked is not None:
            # Create a tensor of patch (row, col) coordinates.
            coords = []
            for i in range(self.grid_h):
                for j in range(self.grid_w):
                    coords.append((i, j))
            coords = torch.tensor(coords, dtype=torch.float)
            # Compute Euclidean distance from center of grid.
            center = torch.tensor([(self.grid_h - 1) / 2, (self.grid_w - 1) / 2], dtype=torch.float)
            self.register_buffer('patch_distances', torch.sqrt(((coords - center) ** 2).sum(dim=1)))
            sorted_indices = torch.argsort(self.patch_distances)  # ascending order of distance
            self.register_buffer('sorted_indices', sorted_indices)
            self.num_patches = num_patches
     
    def patch_to_latent(self, x):
        
        #print(x.size())
        x = self.to_latent(x)
        return self.mlp_head(x)

    def forward(self, img):
        # Extract patch embeddings: result has shape (b, num_patches, dim)
        x = self.to_patch_embedding(img)
        #print(x.size())

        # If masking is enabled, select 50% of patches based on distance.
        if self.masked is not None:
            num_total = self.num_patches
            num_mask = num_total // 2  # drop 50%
            if self.masked == 'center':
                mask_indices = self.sorted_indices[:num_mask]
            elif self.masked == 'outer':
                mask_indices = self.sorted_indices[-num_mask:]
            else:
                raise ValueError("masked must be either None, 'center', or 'outer'")
            full_mask = torch.ones(num_total, dtype=torch.bool, device=x.device)
            full_mask[mask_indices] = False  # False indicates masked patch
            x = x[:, full_mask, :]  # keep only unmasked patches

            # Adjust positional embeddings: retain cls token plus unmasked patch positions.
            pos_emb = torch.cat([self.pos_embedding[:, :1],
                                 self.pos_embedding[:, 1:][:, full_mask]], dim=1)
        else:
            pos_emb = self.pos_embedding

        b = x.shape[0]
        # Append cls token.
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_emb
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return x  # returns : [B, num_total, dim], [B, dim]