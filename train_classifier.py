from pathlib import Path
import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models, transforms

def build_dataloaders(data_dir: str, batch_size: int, val_ratio: float = 0.2, seed: int = 42):
    tf_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(root=data_dir, transform=tf_train)
    num_val = max(1, int(len(full) * val_ratio))
    num_train = len(full) - num_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [num_train, num_val], generator=gen)

    val_ds.dataset = datasets.ImageFolder(root=data_dir, transform=tf_val)
    class_names = full.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return train_loader, val_loader, class_names

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/category", help="クラス別フォルダの親ディレクトリ")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-rato", type=float, default=0.2)
    args = p.parse_args()

    data_dir = args.data_dir
    out_path = Path(__file__).parent / "models" / "cat_classifier.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir, batch_size=args.batch_size, val_ratio=args.val_ratio
    )

    # ResNet18 を転移学習
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, len(class_names))
    base = base.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(base, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(base, val_loader, criterion, device)
        print(f"[{epoch:02d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = base.state_dict()

    if best_state is None:
        best_state = base.state_dict()

    torch.save({
        "model_state": best_state,
        "class_names": class_names,
    }, out_path)

    print(f"✅ Saved model to: {out_path}  (best val acc={best_acc:.3f})")
    print(f"✅ Classes: {class_names}")

if __name__ == "__main__":
    main()
