import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import transformer_model as tm

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

@dataclass
class Config:
    batch_size = 32
    image_size = 32 
    patch_size = 4 # tokens of 4x4 pixels
    n_embd = 32
    n_channels = 1
    normalize_shape = (0.5)
    n_blocks = 2

    image_size = 32
    n_classes = 10

class PatchEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        c = config.n_channels
        p = config.image_size // config.patch_size # image size(32) // patch size(4) = 8 patches
        self.proj = nn.Linear(c*p*p, config.n_embd)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.config.patch_size

        x = x.unfold(2, p, p) # (b, c, h/p, p, w)
        x = x.unfold(3, p, p) # (b, c, h/p, w/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous() # (b, h/p, w/p, c, p, p)
        x = x.view(b, -1, c*p*p) # (b, n, c*p*p) where n=(h/p) * (w/p)
        x = self.proj(x) # (b, n, n_embd), each batch is n patches of n_embd dimensions
        return x
    
class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        n = (config.image_size // config.patch_size) ** 2
        d = config.n_embd

        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, n+1, d) * 0.02)

        self.transformer_blocks = nn.Sequential(
            *[tm.TransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.ln = tm.MyLayerNorm(dim=d)
        self.head = nn.Linear(d, config.n_classes)

    def forward(self, x):
        b, c, w, h = x.shape
        patches = self.patch_embedding(x) # (b, n, d)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1) # (b, n+1, d)
        x = x + self.pos_emb[:, :x.size(1):, :] # (b, n+1, d)

        x = self.transformer_blocks(x)
        x = self.ln(x[:, 0, :])
        logits = self.head(x)
        return logits
    
def test_modules(config):
    pass

def main():
    config = Config()
    test_modules(config)

if __name__ == '__main__':
    main()