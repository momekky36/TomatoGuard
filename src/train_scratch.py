"""
train_scratch.py
-----------------
Trains the SimpleCNN baseline fully from scratch (no pretrained weights
needed - every layer learns from random init). Used when ImageNet weights
aren't reachable (restricted network) or as a "what does transfer learning
actually buy you" comparison point against train.py.

Saves:
  - outputs_scratch/best_model.pt
  - outputs_scratch/training_curves.png
  - outputs_scratch/class_names.json
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import build_dataloaders
from model_scratch import build_scratch_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--out-dir", default="../outputs_scratch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | from-scratch SimpleCNN | image_size={args.image_size}", flush=True)

    train_loader, val_loader, _, classes = build_dataloaders(
        args.data_dir, args.batch_size, num_workers=1, image_size=args.image_size
    )
    json.dump(classes, open(os.path.join(args.out_dir, "class_names.json"), "w"), indent=2)

    model = build_scratch_model(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable params: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1, patience_ctr = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.3f} acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f} f1={val_f1:.3f} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> new best model saved (val_f1={best_f1:.3f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).", flush=True)
                break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150)
    print(f"\nBest val macro-F1: {best_f1:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
