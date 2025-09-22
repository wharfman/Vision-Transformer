import torch
import torch.nn as nn
import torchvision.transforms as T
import transformer_model as tm
import logging
import sys

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from dataclasses import dataclass

# Set the device and seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1337
torch.manual_seed(seed)
print(f"Device {device}")
print(f"Seed {seed}")


logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt")
    ]
)

@dataclass
class Config:
    batch_size = 128
    image_size = 32 
    patch_size = 4 # tokens of 4x4 pixels
    normalize_shape = (0.5, 0.5, 0.5)
    learning_rate = 1e-3

    p_dropout = 0.2
    p_train_split = 0.9
    
    n_embd = 32
    n_channels = 3
    n_heads = 8
    n_blocks = 3
    n_epochs = 300
    n_classes = 10
    

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

        # Create patches: (b, c, h/p, p, w) -=> (b, c, h/p, w/p, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p) 
        # Rearrange so that patches are the sequence dimension: (b, h/p, w/p, c, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous() 
        # Flatten the patches: (b, n, c*p*p) where n=(h/p) * (w/p)
        x = x.view(b, -1, c*p*p) 
        # Project patches to the embedding dimension: (b, n, n_embd)
        x = self.proj(x) 
        return x
    
class ViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        n = (config.image_size // config.patch_size) ** 2
        d = config.n_embd

        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, n+1, d) * 0.02)

        # Create transformer block sequences
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
        x = x + self.pos_emb[:, :x.size(1), :] # Add positional embedding (b, n+1, d)

        x = self.transformer_blocks(x)
        x = self.ln(x[:, 0, :])
        logits = self.head(x)
        return logits
    

def train_test_model(config):
    dataset = CIFAR10(
        root="../data/",
        download=True,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)), # Resize dataset image to config parameter (32x32)
            T.RandomCrop(config.image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15), # Rotate image by 15 degrees
            T.ToTensor(), # Tensor in range 0,1
            T.Normalize(config.normalize_shape, config.normalize_shape), # Standardize pixels from 0,1 to -1,1
        ])
    )
    # Split dataset into training and validation sets
    train_split = int(config.p_train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    train_dataloader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = ViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.n_epochs + 1):
        logging.info(f"Epoch {epoch}/{config.n_epochs}")

        # Training 
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            # Clip gradient to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        logging.info(f" --> Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        logging.info(f" --> Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f" Learning Rate: {current_lr:.6f}\n")


def main():
    config = Config()
    train_test_model(config)

if __name__ == '__main__':
    main()