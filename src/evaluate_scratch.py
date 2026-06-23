"""
evaluate_scratch.py
--------------------
Same as evaluate.py but for the from-scratch SimpleCNN checkpoint
(model_scratch.py, image_size=96).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders
from model_scratch import build_scratch_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--out-dir", default="../outputs_scratch")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or os.path.join(args.out_dir, "best_model.pt")
    classes = json.load(open(os.path.join(args.out_dir, "class_names.json")))

    _, _, test_loader, _ = build_dataloaders(args.data_dir, image_size=args.image_size)

    model = build_scratch_model(num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Tomato Leaf Disease Test Set (from-scratch CNN)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    print(f"\nSaved confusion matrix + report to {args.out_dir}")


if __name__ == "__main__":
    main()
