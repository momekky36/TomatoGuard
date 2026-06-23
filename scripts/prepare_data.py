"""
prepare_data.py
----------------
Downloads (via sparse git checkout) the Tomato subset of the open-source
PlantVillage leaf-disease dataset and splits it into train/val/test folders
in ImageFolder format:

    data/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg

Usage:
    python prepare_data.py --max-per-class 99999   # use the full dataset
    python prepare_data.py --max-per-class 200      # quick subset for a fast smoke test
"""
import argparse
import os
import random
import shutil
import subprocess
import sys

SOURCE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
RAW_SUBDIR = "raw/color"
CLASS_PREFIX = "Tomato___"


def clone_source(tmp_dir: str):
    if os.path.exists(tmp_dir):
        print(f"[skip] {tmp_dir} already exists")
        return
    print("Cloning PlantVillage-Dataset (sparse checkout, color images only)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", SOURCE_REPO, tmp_dir],
        check=True,
    )
    subprocess.run(["git", "-C", tmp_dir, "sparse-checkout", "set", RAW_SUBDIR], check=True)


def split_dataset(raw_dir: str, out_dir: str, max_per_class: int, seed: int = 42):
    random.seed(seed)
    class_dirs = sorted(d for d in os.listdir(raw_dir) if d.startswith(CLASS_PREFIX))
    if not class_dirs:
        sys.exit(f"No '{CLASS_PREFIX}*' folders found in {raw_dir}")

    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    for split in splits:
        for cls in class_dirs:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    summary = {}
    for cls in class_dirs:
        files = os.listdir(os.path.join(raw_dir, cls))
        random.shuffle(files)
        files = files[:max_per_class]

        n = len(files)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])
        buckets = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, fl in buckets.items():
            for f in fl:
                shutil.copy(
                    os.path.join(raw_dir, cls, f),
                    os.path.join(out_dir, split, cls, f),
                )
        summary[cls] = {k: len(v) for k, v in buckets.items()}

    print("\nClass distribution:")
    print(f"{'class':45s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for cls, d in summary.items():
        print(f"{cls:45s} {d['train']:6d} {d['val']:6d} {d['test']:6d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp-dir", default="temp_repo")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-per-class", type=int, default=99999,
                         help="Cap images per class (use a small number for a fast smoke test)")
    args = parser.parse_args()

    clone_source(args.tmp_dir)
    raw_dir = os.path.join(args.tmp_dir, RAW_SUBDIR)
    split_dataset(raw_dir, args.out_dir, args.max_per_class)
    print("\nDone. Data ready in:", args.out_dir)
