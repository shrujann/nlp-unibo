"""
Parse the downloaded Nutrition5k sample and upload it to the local Phoenix instance
as a named dataset.  Skips the upload if the dataset already exists (persistence).

Run once after download_dataset.py:
  python experiments/upload_dataset.py
"""
import csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client

from experiments.utils import ensure_phoenix

load_dotenv()
ensure_phoenix()

DATA_DIR = Path(__file__).parent / "data" / "nutrition5k"
DATASET_NAME = "nutrition5k-50"


def _parse_ingredients(row: dict) -> list[dict]:
    ingredients = []
    i = 1
    while f"ingr_{i}_name" in row:
        name = (row.get(f"ingr_{i}_name") or "").strip()
        grams = float(row.get(f"ingr_{i}_grams") or 0)
        calories = float(row.get(f"ingr_{i}_calories") or 0)
        if name and grams > 0:
            ingredients.append({"name": name, "grams": grams, "calories": round(calories, 1)})
        i += 1
    return ingredients


def upload():
    client = Client()

    # Don't re-upload if already stored in Phoenix
    try:
        dataset = client.datasets.get_dataset(dataset=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already in Phoenix — reusing.")
        return dataset
    except Exception:
        pass

    sample_path = DATA_DIR / "sample.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"{sample_path} not found — run download_dataset.py first."
        )

    records = []
    with open(sample_path, newline="") as f:
        for row in csv.DictReader(f):
            dish_id = row["dish_id"]
            records.append(
                {
                    "image_path": str(DATA_DIR / "images" / dish_id / "rgb.png"),
                    "dish_id": dish_id,
                    "total_calories": round(float(row["total_calories"]), 1),
                    "ingredients": _parse_ingredients(row),
                }
            )

    df = pd.DataFrame(records)
    dataset = client.datasets.create_dataset(
        name=DATASET_NAME,
        dataframe=df,
        input_keys=["image_path", "dish_id"],
        output_keys=["total_calories", "ingredients"],
        dataset_description="50 Nutrition5k dishes for calorie estimation evaluation",
    )
    print(f"Uploaded {len(records)} examples as dataset '{DATASET_NAME}'.")
    return dataset


if __name__ == "__main__":
    upload()
