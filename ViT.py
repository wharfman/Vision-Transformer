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
    n_heads = 2
    normalize_shape = (0.5)
    n_blocks = 2
    p_dropout = 0.2

    n_epochs = 3
    image_size = 32
    n_classes = 10
    p_train_split = 0.9

class PatchEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.n_channels
        p = config.patch_size 
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

def train_test_model(config):
    dataset = MNIST(
        root="..",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(config.normalize_shape, config.normalize_shape), # Standardize pixels from 0,1 to -1,1
        ])
    )
    train_split = int(config.p_train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = ViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        with tqdm(train_dataloader, desc="Training", unit="batch") as pbar:
            model.train()
            train_loss = 0.0
            train_correct = 0.0
            train_total = 0
            for inputs, labels in pbar:
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                pbar.set_postfix(loss=loss.item())
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        tqdm.write(f"Train Loss {train_epoch_loss:.4f} Train Acc {train_epoch_acc:.2f}")
        
        with tqdm(test_dataloader, desc="Validation", unit="batch") as pbar:
            model.eval()
            val_loss = 0.0
            val_correct = 0.0
            val_total = 0
            for inputs, labels in pbar:
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                pbar.set_postfix(loss=loss.item())
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        tqdm.write(f"Val loss {val_epoch_loss:.4f} Val Acc {val_epoch_acc:.2f}")


def main():
    config = Config()
    train_test_model(config)

if __name__ == '__main__':
    main()