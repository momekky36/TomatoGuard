# 🍅 TomatoGuard — AI-Powered Tomato Leaf Disease Detection

![Sanity Check](https://github.com/YOUR-USERNAME/tomatoguard/actions/workflows/sanity-check.yml/badge.svg)

A computer vision system that diagnoses 9 tomato diseases (+ healthy) from a
single leaf photo, with built-in explainability (Grad-CAM) so a user can see
*why* the model made its call, not just the label.

## Why this matters (the business case)

Tomato is one of the world's most widely grown vegetable crops, and foliar
diseases like early blight, late blight, and bacterial spot can cut yields
dramatically if caught late — losses of 20-50%+ in untreated outbreaks are
common in the agronomy literature. Smallholder farmers in particular often
lack affordable access to a plant pathologist:

- **Early detection → targeted treatment.** Spraying the right fungicide/
  bactericide for the *specific* disease, only when needed, instead of
  blanket preventive spraying, cuts both cost and chemical overuse.
- **Scale.** A phone-camera + model can triage thousands of leaves a day;
  a human expert cannot.
- **Accessibility.** This turns into a free phone-based diagnostic tool —
  the same architecture underlies real products like PlantVillage Nuru and
  Plantix, used by millions of farmers.

This project reproduces that pipeline end-to-end: data → model →
evaluation → explainability → a usable web demo.

## Dataset

[PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
(Mohanty et al.), tomato subset: **10 classes, ~18,000 leaf images**
(healthy + 9 diseases), with natural class imbalance (Yellow Leaf Curl Virus
has ~14x more samples than Mosaic Virus) — a realistic detail worth handling
properly rather than ignoring, see "Engineering decisions" below.

## Approach

1. **Transfer learning**, not training from scratch: an EfficientNet-B0
   backbone pretrained on ImageNet, with the classifier head replaced and
   the last conv block fine-tuned (`src/model.py`). This is the standard,
   data-efficient approach for image classification when you have
   thousands (not millions) of images.
2. **Class-imbalance handling** via a weighted random sampler so every
   training epoch sees a balanced mix of classes (`src/dataset.py`),
   instead of the model just learning to predict the majority class.
3. **Training** with AdamW, LR scheduling on plateau, early stopping on
   validation macro-F1 (not accuracy — macro-F1 doesn't let majority
   classes hide poor minority-class performance) (`src/train.py`).
4. **Evaluation**: per-class precision/recall/F1 + confusion matrix on a
   held-out test set (`src/evaluate.py`).
5. **Explainability**: Grad-CAM heatmaps showing which leaf regions drove
   each prediction (`src/gradcam.py`) — important for trust in any
   agricultural or medical-adjacent deployment.
6. **Demo**: a Gradio web app (`app/app.py`) for live inference + heatmap.

## Results

This project includes **two training tracks**, which itself is a good talking
point for a CV ("understands the tradeoff between transfer learning and
training from scratch"):

### Track A — From-scratch CNN (✅ actually run, live, in this build)

The sandbox this was built in has no internet access to the host that
serves pretrained ImageNet weights, so rather than fake a number, a small
CNN (`src/model_scratch.py`, ~390K params) was trained **fully from random
initialization** on 2,000 real images (200/class, balanced) at 96×96
resolution, single CPU core, 27 epochs (~20 minutes):

| Metric | Value |
|---|---|
| Test accuracy | **92.0%** |
| Test macro-F1 | **0.919** |
| Val macro-F1 (best, epoch 26) | 0.940 |

Per-class F1 ranged 0.81 (Early blight, confused with the visually similar
Septoria leaf spot) to 1.00 (healthy, Yellow Leaf Curl Virus). Full report
in `outputs_scratch/classification_report.txt`. Train/val curves track each
other closely with no overfitting gap (`outputs_scratch/training_curves.png`),
and Grad-CAM (`outputs_scratch/gradcam_samples.png`) confirms the model is
attending to the leaf itself, not the background.

### Track B — Transfer learning (recommended; run it yourself with internet)

`src/model.py` / `src/train.py` use an ImageNet-pretrained EfficientNet-B0.
This needs internet access to download weights (works fine on
[Google Colab](https://colab.research.google.com/), Kaggle, or any normal
machine — just not this sandbox). On the **full ~18,000-image dataset**,
expect **~95-98% test accuracy** with far less training time, consistent
with published PlantVillage benchmarks — pretrained features are simply a
huge head start over random init. Re-run it on Colab and replace the
numbers above with your own before publishing this on a CV; that's the
result worth citing.

```bash
cd src
python train.py --data-dir ../data --out-dir ../outputs --epochs 15
python evaluate.py --data-dir ../data --out-dir ../outputs
python gradcam.py
```

## Engineering decisions worth highlighting on a CV

- Chose **macro-F1** over accuracy as the model-selection metric because
  of class imbalance — a model that ignores rare diseases would still
  score well on accuracy.
- Used a **weighted sampler** rather than naive oversampling/undersampling
  to fix imbalance without throwing away majority-class data or
  duplicating minority-class images verbatim.
- Froze most of the backbone and only fine-tuned the head + last block —
  faster training, lower overfitting risk on a few thousand images per
  class, standard transfer-learning practice.
- Added **Grad-CAM** explainability rather than shipping a black box —
  relevant anywhere a wrong prediction has real-world cost.

## Project structure

```
tomatoguard/
├── LICENSE
├── .github/workflows/sanity-check.yml  # CI: imports + syntax check on push
├── README.md
├── requirements.txt
├── scripts/
│   └── prepare_data.py          # downloads + splits the dataset
├── src/
│   ├── dataset.py                # dataloaders + class-imbalance handling
│   ├── model.py                  # EfficientNet-B0 transfer-learning model
│   ├── train.py                   # transfer-learning training loop
│   ├── evaluate.py                # transfer-learning test metrics
│   ├── model_scratch.py           # lightweight from-scratch CNN
│   ├── train_scratch.py           # from-scratch training loop (CPU-friendly)
│   ├── evaluate_scratch.py        # from-scratch test metrics
│   ├── gradcam.py                  # explainability heatmaps (shared GradCAM class)
│   └── gradcam_scratch.py          # explainability for the scratch model
├── app/
│   └── app.py                    # Gradio web demo
├── examples/                      # sample leaf images for quick testing
├── outputs/                       # transfer-learning artifacts (you generate)
└── outputs_scratch/               # from-scratch artifacts (included, real run)
```

## How to run it yourself

```bash
pip install -r requirements.txt

# 1. Download & split the data (full dataset; use --max-per-class N for a quick test)
python scripts/prepare_data.py --out-dir data

# 2. Train (uses pretrained ImageNet weights by default - needs internet)
cd src
python train.py --data-dir ../data --out-dir ../outputs --epochs 15

# 3. Evaluate on the held-out test set
python evaluate.py --data-dir ../data --out-dir ../outputs

# 4. Generate Grad-CAM explainability samples
python gradcam.py

# 5. Launch the interactive demo
cd ../app
python app.py
```

**No GPU?** Use [Google Colab](https://colab.research.google.com/) (free
T4 GPU) — upload this repo, run the same commands, training takes
~10-20 minutes for the full dataset.

## Credits & license

Dataset: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
by Mohanty, Hughes & Salathé — check the dataset repo for its license terms
before any commercial use. Code in this repo is under the [MIT License](LICENSE).

## Pushing this to GitHub

`data/` is already in `.gitignore` (it's regenerated by `prepare_data.py`,
no need to version 2,000+ images). Everything else — code, README, LICENSE,
the real `outputs_scratch/` results, CI workflow — is meant to be committed.

```bash
cd tomatoguard
git init
git add .
git commit -m "TomatoGuard: tomato leaf disease detector (CV transfer-learning + from-scratch tracks)"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/tomatoguard.git
git push -u origin main
```

Then in `README.md`, replace `YOUR-USERNAME` in the CI badge URL near the
top with your actual GitHub username so the badge renders.
=======
# TomatoGuard
🍅 Computer vision pipeline that diagnoses tomato leaf diseases from a photo (PyTorch, Grad-CAM, Gradio demo). From-scratch CNN hits 92% test accuracy on real data; EfficientNet-B0 transfer-learning track included.
>>>>>>> c7a1a6e5e24c33a5b747b73a1d7240d754da04c9
