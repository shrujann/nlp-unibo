"""
Parse the downloaded Nutrition5k sample and upload it to the local Phoenix instance
as a named dataset.  Skips the upload if the dataset already exists (persistence).

Run once after download_dataset.py:
  python -m experiments.upload_dataset
"""
import csv
from pathlib import Path

import pandas as pd
import phoenix as px
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data" / "nutrition5k"
DATASET_NAME = "nutrition5k-50"


def _parse_ingredients(row: dict) -> list[dict]:
    n = int(row.get("num_ingrs") or 0)
    ingredients = []
    for i in range(1, n + 1):
        name = row.get(f"ingr_{i}_name", "").strip()
        grams = float(row.get(f"ingr_{i}_grams") or 0)
        calories = float(row.get(f"ingr_{i}_calories") or 0)
        if name and grams > 0:
            ingredients.append({"name": name, "grams": grams, "calories": round(calories, 1)})
    return ingredients


def upload() -> px.Dataset:
    client = px.Client()

    # Don't re-upload if already stored in Phoenix
    try:
        existing_names = {ds.name for ds in client.list_datasets()}
        if DATASET_NAME in existing_names:
            print(f"Dataset '{DATASET_NAME}' already in Phoenix — reusing.")
            return client.get_dataset(name=DATASET_NAME)
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
                    # inputs (what the task function receives)
                    "image_path": str(DATA_DIR / "images" / dish_id / "rgb.png"),
                    "dish_id": dish_id,
                    # expected outputs (what evaluators compare against)
                    "total_calories": round(float(row["total_calories"]), 1),
                    "ingredients": _parse_ingredients(row),
                }
            )

    df = pd.DataFrame(records)
    dataset = client.upload_dataset(
        dataframe=df,
        name=DATASET_NAME,
        input_keys=["image_path", "dish_id"],
        output_keys=["total_calories", "ingredients"],
    )
    print(f"Uploaded {len(records)} examples as dataset '{DATASET_NAME}'.")
    return dataset


if __name__ == "__main__":
    upload()
