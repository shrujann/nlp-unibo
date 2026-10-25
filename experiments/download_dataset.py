"""
Download 50 Nutrition5k dishes from Google Cloud Storage.

Requires gsutil (part of Google Cloud SDK):
  https://cloud.google.com/storage/docs/gsutil_install

Run once before upload_dataset.py:
  python -m experiments.download_dataset
"""
import csv
import random
import subprocess
from pathlib import Path

SEED = 42
N = 50
BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"
DATA_DIR = Path(__file__).parent / "data" / "nutrition5k"


def _gsutil(*args: str) -> None:
    subprocess.run(["gsutil", *args], check=True)


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch the metadata CSV (cafe1 — ~3 k dishes)
    meta_path = DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        print("Downloading metadata CSV…")
        _gsutil("cp", f"{BUCKET}/metadata/dish_metadata_cafe1.csv", str(meta_path))

    # 2. Sample N dishes with a fixed seed for reproducibility
    with open(meta_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(SEED)
    sample = rng.sample(rows, N)

    # 3. Save the sample so upload_dataset.py knows which rows to use
    sample_path = DATA_DIR / "sample.csv"
    with open(sample_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sample)

    # 4. Download the overhead RGB image for each sampled dish
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    for i, row in enumerate(sample, 1):
        dish_id = row["dish_id"]
        img_path = images_dir / dish_id / "rgb.png"
        if img_path.exists():
            print(f"  [{i}/{N}] {dish_id} already exists, skipping")
            continue
        img_path.parent.mkdir(exist_ok=True)
        print(f"  [{i}/{N}] downloading {dish_id}")
        _gsutil(
            "cp",
            f"{BUCKET}/imagery/realsense_overhead/{dish_id}/rgb.png",
            str(img_path),
        )

    print(f"\nDone — {N} dishes saved to {DATA_DIR}")


if __name__ == "__main__":
    download()
