"""
Download 50 Nutrition5k dishes directly from Google Cloud Storage over HTTPS.
No gsutil or Google Cloud SDK required — uses httpx (already installed).

The metadata CSV has no header row; columns are positional:
  0: dish_id  1: total_calories  2: total_mass  3: total_fat
  4: total_carb  5: total_protein  6: num_ingrs
  7+: ingr_N_id, ingr_N_name, ingr_N_grams, ingr_N_calories,
      ingr_N_fat, ingr_N_carb, ingr_N_protein  (7 fields each)

Run from the project root:
  python experiments/download_dataset.py
"""
import asyncio
import csv
import io
import random
from pathlib import Path

import httpx

SEED = 42
N = 50
OVERSAMPLE = N * 3          # download 3× candidates to absorb 404s
GCS_BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
DATA_DIR = Path(__file__).parent / "data" / "nutrition5k"

# Fixed column positions (0-indexed); ingredients start at 6 with no num_ingrs column
COL_DISH_ID  = 0
COL_CALORIES = 1
COL_MASS     = 2
COL_FAT      = 3
COL_CARB     = 4
COL_PROTEIN  = 5
INGR_START   = 6
INGR_FIELDS  = ["id", "name", "grams", "calories", "fat", "carb", "protein"]  # 7 per ingredient
INGR_STRIDE  = len(INGR_FIELDS)


def _num_ingrs(row: list) -> int:
    return (len(row) - INGR_START) // INGR_STRIDE


def _col_headers(max_ingrs: int) -> list[str]:
    fixed = ["dish_id", "total_calories", "total_mass", "total_fat",
             "total_carb", "total_protein"]
    ingr = [f"ingr_{i}_{f}" for i in range(1, max_ingrs + 1) for f in INGR_FIELDS]
    return fixed + ingr


async def _fetch(client: httpx.AsyncClient, url: str, dest: Path) -> bool:
    try:
        r = await client.get(url, follow_redirects=True, timeout=30.0)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


async def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        # 1. Fetch the metadata CSV
        meta_path = DATA_DIR / "metadata.csv"
        if not meta_path.exists():
            print("Downloading metadata CSV…")
            r = await client.get(
                f"{GCS_BASE}/metadata/dish_metadata_cafe1.csv",
                follow_redirects=True,
                timeout=60.0,
            )
            r.raise_for_status()
            meta_path.write_bytes(r.content)

        # 2. Parse without header — every row is a dish
        with open(meta_path, newline="") as f:
            all_rows = [r for r in csv.reader(f) if r and r[0].startswith("dish_")]
        print(f"Metadata: {len(all_rows)} dishes total")

        rng = random.Random(SEED)
        candidates = rng.sample(all_rows, min(OVERSAMPLE, len(all_rows)))

        # 3. Download images concurrently; collect which ones succeed
        images_dir = DATA_DIR / "images"
        images_dir.mkdir(exist_ok=True)
        sem = asyncio.Semaphore(10)

        async def fetch_dish(row: list) -> bool:
            dish_id = row[COL_DISH_ID]
            img_path = images_dir / dish_id / "rgb.png"
            if img_path.exists():
                return True
            img_path.parent.mkdir(exist_ok=True)
            async with sem:
                ok = await _fetch(
                    client,
                    f"{GCS_BASE}/imagery/realsense_overhead/{dish_id}/rgb.png",
                    img_path,
                )
            if not ok and img_path.parent.exists():
                import shutil
                shutil.rmtree(img_path.parent, ignore_errors=True)
            return ok

        print(f"Trying {len(candidates)} candidates to get {N} successes…")
        results = await asyncio.gather(*[fetch_dish(r) for r in candidates])

        sample = [row for row, ok in zip(candidates, results) if ok][:N]
        print(f"Downloaded {sum(results)}/{len(candidates)} images — keeping {len(sample)}")

        # Remove any extra images that were downloaded but aren't in the final sample
        import shutil as _shutil
        kept_ids = {row[COL_DISH_ID] for row in sample}
        for dish_dir in images_dir.iterdir():
            if dish_dir.is_dir() and dish_dir.name not in kept_ids:
                _shutil.rmtree(dish_dir, ignore_errors=True)

        if len(sample) < N:
            print(f"Warning: only {len(sample)} images available (wanted {N})")

        # 4. Write sample.csv WITH explicit headers so upload_dataset.py can use named columns
        max_ingrs = max(_num_ingrs(row) for row in sample)
        headers = _col_headers(max_ingrs)

        sample_path = DATA_DIR / "sample.csv"
        with open(sample_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in sample:
                padded = row + [""] * max(0, len(headers) - len(row))
                writer.writerow(padded[: len(headers)])

        print(f"\nDone — {len(sample)} dishes saved to {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(download())
