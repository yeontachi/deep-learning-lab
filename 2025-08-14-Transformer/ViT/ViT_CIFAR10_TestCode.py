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

# data

def get_dataloaders(cfg: Config):
    # CIFAR-10 통계
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tfms = transforms.Compose([
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader

# Train / Eval
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, leave=False)
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
        pbar.set_description(f"train loss {total_loss/total:.4f} acc {correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main(cfg: Config):
    set_seed(cfg.seed)
    train_loader, test_loader = get_dataloaders(cfg)

    model = ViT(cfg).to(cfg.device)
    print(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        # cosine + warmup
        lr = cosine_warmup_lr(epoch, cfg.lr, cfg.epochs, cfg.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, cfg.device, scaler)
        val_loss, val_acc = evaluate(model, test_loader, cfg.device)

        print(f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
              f"lr {lr:.6f} | train {train_loss:.4f}/{train_acc:.4f} | valid {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "vit_cifar10_best.pth")
            print(f"  ↳ Saved best (acc={best_acc:.4f})")

    print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main(CFG)