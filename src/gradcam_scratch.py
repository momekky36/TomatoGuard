"""
gradcam_scratch.py
-------------------
Grad-CAM for the from-scratch SimpleCNN checkpoint (image_size=96).
Target layer is the last conv block's ReLU output (features[14]), the last
point with spatial resolution before the global-average-pool head.
"""
import json
import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import get_transforms
from model_scratch import build_scratch_model
from gradcam import GradCAM


def main(data_dir="../data", out_dir="../outputs_scratch", n_samples=6, image_size=96):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = json.load(open(os.path.join(out_dir, "class_names.json")))
    model = build_scratch_model(num_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.to(device).eval()

    target_layer = model.features[14]  # last ReLU before the global-avg-pool
    cam_extractor = GradCAM(model, target_layer)
    _, eval_tf = get_transforms(image_size)

    test_root = os.path.join(data_dir, "test")
    sample_classes = random.sample(classes, min(n_samples, len(classes)))

    fig, axes = plt.subplots(len(sample_classes), 2, figsize=(6, 3 * len(sample_classes)))
    for row, cls in enumerate(sample_classes):
        cls_dir = os.path.join(test_root, cls)
        img_name = random.choice(os.listdir(cls_dir))
        img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
        tensor = eval_tf(img).unsqueeze(0).to(device)

        cam, pred_idx = cam_extractor(tensor)

        img_resized = img.resize((image_size, image_size))
        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title(f"True: {cls}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_resized)
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[row, 1].set_title(f"Pred: {classes[pred_idx]}", fontsize=8)
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradcam_samples.png"), dpi=150)
    print(f"Saved Grad-CAM grid to {out_dir}/gradcam_samples.png")


if __name__ == "__main__":
    main()
