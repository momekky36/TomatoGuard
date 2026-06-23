"""
gradcam.py
----------
Generates Grad-CAM heatmaps showing WHICH pixels of a leaf image the model
relied on for its prediction. This is the difference between "it works" and
"I understand why it works" - a strong interpretability signal for a CV.

Produces outputs/gradcam_samples.png: a grid of (original | heatmap overlay)
pairs, one row per class, for a handful of random test images.
"""
import json
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_transforms
from model import build_model


class GradCAM:
    """Minimal Grad-CAM: hooks the last conv block, backprops the predicted
    class's logit, and weights the activation maps by their gradients."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor):
        output = self.model(input_tensor)
        pred_idx = output.argmax(1).item()

        self.model.zero_grad()
        output[0, pred_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_idx


def main(data_dir="../data", out_dir="../outputs", n_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = json.load(open(os.path.join(out_dir, "class_names.json")))
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.to(device).eval()

    cam_extractor = GradCAM(model, model.features[-1])
    _, eval_tf = get_transforms()

    test_root = os.path.join(data_dir, "test")
    sample_classes = random.sample(classes, min(n_samples, len(classes)))

    fig, axes = plt.subplots(len(sample_classes), 2, figsize=(6, 3 * len(sample_classes)))
    for row, cls in enumerate(sample_classes):
        cls_dir = os.path.join(test_root, cls)
        img_name = random.choice(os.listdir(cls_dir))
        img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
        tensor = eval_tf(img).unsqueeze(0).to(device)

        cam, pred_idx = cam_extractor(tensor)

        img_resized = img.resize((224, 224))
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
