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
    
