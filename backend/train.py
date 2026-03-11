"""
Training Script for AerialDet-Hybrid
=====================================
Supports: DOTA, VEDAI, VisDrone datasets (structure: images + YOLO-style labels)
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path

from models.hybrid_model import AerialDetHybrid, AerialDetLoss


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class AerialDataset(Dataset):
    """
    Generic aerial detection dataset.
    Expects:
      root/
        images/  *.jpg or *.png
        labels/  *.txt  (YOLO format: cls cx cy w h per line, normalized)
    """
    def __init__(self, root: str, img_size: int = 640,
                 augment: bool = True):
        self.img_size = img_size
        self.augment  = augment

        img_dir = Path(root) / 'images'
        lbl_dir = Path(root) / 'labels'
        self.samples = sorted([
            (str(p), str(lbl_dir / p.with_suffix('.txt').name))
            for p in img_dir.glob('*.jpg')
        ] + [
            (str(p), str(lbl_dir / p.with_suffix('.txt').name))
            for p in img_dir.glob('*.png')
        ])

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        boxes, labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        labels.append(int(cls))
                        boxes.append([cx, cy, w, h])

        return {
            'image':  img,
            'boxes':  torch.tensor(boxes,  dtype=torch.float32)
                      if boxes  else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long)
                      if labels else torch.zeros(0, dtype=torch.long),
        }


def collate_fn(batch):
    images  = torch.stack([b['image']  for b in batch])
    targets = [{'boxes': b['boxes'], 'labels': b['labels']} for b in batch]
    return images, targets


# ─────────────────────────────────────────────
#  Cosine LR Scheduler
# ─────────────────────────────────────────────
def cosine_lr(optimizer, epoch, total_epochs, warmup=5,
              lr_min=1e-6, lr_max=1e-4):
    if epoch < warmup:
        lr = lr_max * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / (total_epochs - warmup)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────
def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[AerialDet] Training on: {device}")

    # Dataset & DataLoader
    train_ds = AerialDataset(cfg['data_root'], cfg['img_size'], augment=True)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'],
                          shuffle=True, num_workers=cfg['num_workers'],
                          collate_fn=collate_fn, pin_memory=True)

    # Model
    model = AerialDetHybrid(
        num_classes   = cfg['num_classes'],
        num_proposals = cfg['num_proposals'],
        d_model       = cfg['d_model'],
        n_heads       = cfg['n_heads'],
        num_enc_layers= cfg['num_enc_layers'],
        num_stages    = cfg['num_stages'],
        roi_size      = cfg['roi_size'],
        dropout       = cfg['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[AerialDet] Parameters: {total_params/1e6:.1f}M")

    # Loss & Optimizer
    criterion = AerialDetLoss(cfg['num_classes'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = GradScaler(enabled=cfg.get('amp', True))

    best_loss = float('inf')
    os.makedirs(cfg['save_dir'], exist_ok=True)

    for epoch in range(cfg['epochs']):
        model.train()
        lr = cosine_lr(optimizer, epoch, cfg['epochs'],
                       warmup=cfg['warmup_epochs'],
                       lr_max=cfg['lr'])

        epoch_loss = 0.
        t0 = time.time()

        for step, (images, targets) in enumerate(train_dl):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast(enabled=cfg.get('amp', True)):
                stage_outputs = model(images)
                losses = criterion(stage_outputs, targets)
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (step + 1) % cfg['log_every'] == 0:
                print(f"  Epoch {epoch+1}/{cfg['epochs']} "
                      f"Step {step+1}/{len(train_dl)} "
                      f"| loss={loss.item():.4f} "
                      f"| cls={losses['cls'].item():.4f} "
                      f"| l1={losses['l1'].item():.4f} "
                      f"| giou={losses['giou'].item():.4f} "
                      f"| lr={lr:.2e}")

        avg_loss = epoch_loss / len(train_dl)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch+1} done | avg_loss={avg_loss:.4f} | "
              f"time={elapsed:.1f}s")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(cfg['save_dir'], 'best.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': best_loss, 'cfg': cfg}, ckpt_path)
            print(f"  ✓ Saved best checkpoint: {ckpt_path}")

        # Periodic save
        if (epoch + 1) % cfg['save_every'] == 0:
            ckpt_path = os.path.join(cfg['save_dir'], f'epoch_{epoch+1}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'cfg': cfg}, ckpt_path)


# ─────────────────────────────────────────────
#  Default Config
# ─────────────────────────────────────────────
DEFAULT_CFG = {
    'data_root':     'data/aerial',
    'save_dir':      'checkpoints',
    'img_size':      640,
    'num_classes':   15,          # DOTA has 15 categories
    'num_proposals': 100,
    'd_model':       256,
    'n_heads':       8,
    'num_enc_layers': 6,
    'num_stages':    6,
    'roi_size':      7,
    'dropout':       0.1,
    'epochs':        36,
    'warmup_epochs': 3,
    'batch_size':    4,
    'num_workers':   4,
    'lr':            1e-4,
    'wd':            1e-4,
    'grad_clip':     0.1,
    'amp':           True,
    'log_every':     50,
    'save_every':    5,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config (overrides defaults)')
    parser.add_argument('--data',   type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    if args.data:
        cfg['data_root'] = args.data
    if args.epochs:
        cfg['epochs'] = args.epochs

    train(cfg)
