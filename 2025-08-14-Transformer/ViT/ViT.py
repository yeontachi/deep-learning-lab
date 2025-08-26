# Model 
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

@dataclass
class Config:
    image_size: int = 32  # CIFAR-10
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 10
    
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    dropout: float = 0.1
    
    epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
CFG = Config()

# utils
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_warmup_lr(epoch, base_lr, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr*(epoch+1)/warmup_epochs
    
    # cosine decay to 0
    progress = (epoch - warmup_epochs)/max(1, (total_epochs - warmup_epochs))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi*progress))

# Model : ViT(Vision Transformer)

class PatchEmbed(nn.Module):
    # Conv2d(k=P, s=P) 이용한 패치 임베딩
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x): # x: (B,C,H,W)
        x = self.proj(x) # (B,embed_dim,H/P,W/P)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        
    def forward(self, x): # x: (B, N, dim)
        B, N, D = x.shape
        qkv = self.qkv(x) # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] # 각각 (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1))*self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        out = attn @ v # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D) # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
    
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim*mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x +self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class ViT(nn.Module):
    def __init(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # CLS 토큰, 위치 임베딩
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim)*0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, cfg.embed_dim)*0.02)
        self.pos_drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                attn_dropout=cfg.attn_dropout,
                dropout=cfg.dropout
            )
            for  _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        
        # 파라미터 초기화(simple)
        self.apply(self.__init_weights)
    
    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x): # x:(B, 3, H, W)
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat([cls, x], dim = 1)       # (B, N+1, D)
        x = x + self.pos_embed                 # (B, N+1, D)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                      # cls 토큰
        logits = self.head(cls_out)
        return logits
  