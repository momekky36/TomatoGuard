"""
app.py
------
A small Gradio web app that loads a trained checkpoint and lets anyone
upload a tomato leaf photo to get an instant diagnosis + confidence scores
+ a Grad-CAM heatmap explaining the prediction.

Defaults to the from-scratch checkpoint in outputs_scratch/ (the one with
real, achieved results - see README). If you've trained the transfer-
learning model in outputs/ instead, set MODEL_TRACK=transfer below.

Run locally:
    python app.py
Then open the printed local URL, or add `demo.launch(share=True)` to get a
public link (handy for a CV/portfolio link), or deploy for free on
Hugging Face Spaces (see README).
"""
import json
import os
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import get_transforms  # noqa: E402
from gradcam import GradCAM  # noqa: E402

MODEL_TRACK = os.environ.get("MODEL_TRACK", "scratch")  # "scratch" or "transfer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_TRACK == "transfer":
    from model import build_model
    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
    IMG_SIZE = 224
    CLASSES = json.load(open(os.path.join(OUT_DIR, "class_names.json")))
    MODEL = build_model(num_classes=len(CLASSES), pretrained=False)
    MODEL.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE))
    TARGET_LAYER = MODEL.features[-1]
else:
    from model_scratch import build_scratch_model
    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_scratch")
    IMG_SIZE = 96
    CLASSES = json.load(open(os.path.join(OUT_DIR, "class_names.json")))
    MODEL = build_scratch_model(num_classes=len(CLASSES))
    MODEL.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE))
    TARGET_LAYER = MODEL.features[14]

MODEL.to(DEVICE).eval()
CAM = GradCAM(MODEL, TARGET_LAYER)
_, EVAL_TF = get_transforms(IMG_SIZE)


def predict(image: Image.Image):
    if image is None:
        return None, None
    image = image.convert("RGB")
    tensor = EVAL_TF(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    label_scores = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    cam, _ = CAM(tensor)
    img_resized = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap = np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1)  # red=high, blue=low
    overlay = np.clip(0.55 * img_resized + 0.45 * heatmap, 0, 1)

    return label_scores, overlay


with gr.Blocks(title="Tomato Leaf Disease Detector") as demo:
    gr.Markdown(
        "# 🍅 Tomato Leaf Disease Detector\n"
        "Upload a photo of a tomato leaf to get an instant diagnosis across "
        f"{len(CLASSES)} classes, plus a Grad-CAM heatmap showing which part "
        "of the leaf drove the prediction.\n\n"
        f"*Model track: `{MODEL_TRACK}`*"
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Leaf photo")
        with gr.Column():
            out_labels = gr.Label(num_top_classes=3, label="Prediction")
            out_cam = gr.Image(label="Grad-CAM (red = most important region)")
    btn = gr.Button("Diagnose", variant="primary")
    btn.click(predict, inputs=inp, outputs=[out_labels, out_cam])

if __name__ == "__main__":
    demo.launch()
