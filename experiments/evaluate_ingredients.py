"""
Run the LLM-as-a-judge ingredient evaluators on Phoenix-exported JSON files
and produce a clean results CSV for analyze.py.

Usage:
  # Single file — system name inferred from filename
  python -m experiments.evaluate_ingredients experiments/experiment-results/our-app.json

  # Multiple files — all combined into one results.csv
  python -m experiments.evaluate_ingredients experiments/experiment-results/*.json

Output: experiments/data/results.csv
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from experiments.evaluators import _IngredientJudgment, _JUDGE_SYSTEM
from backend.config import CHAT_MODEL, OPENROUTER_BASE_URL


def _judge_sync(pred_ingredients: list, gt_ingredients: list) -> _IngredientJudgment:
    if not gt_ingredients:
        return _IngredientJudgment(matches=[])
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=4096,
        temperature=0,
    ).with_structured_output(_IngredientJudgment)
    gt_names = [i["name"] for i in gt_ingredients]
    pred_names = [i["name"] for i in pred_ingredients]
    return llm.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=f"Ground truth: {json.dumps(gt_names)}\nPredicted: {json.dumps(pred_names)}"),
    ])

RESULTS_CSV = Path(__file__).parent / "data" / "results.csv"
_CACHE_FILE = Path(__file__).parent / "data" / ".judge_cache.json"


def _parse_ingredients(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
    return []


def _cache_key(pred: list, gt: list) -> str:
    blob = json.dumps({
        "pred": sorted(i["name"] for i in pred if isinstance(i, dict) and "name" in i),
        "gt": sorted(i["name"] for i in gt if isinstance(i, dict) and "name" in i),
    }, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _system_name(path: Path) -> str:
    return path.stem  # "our-app", "gpt4o-mini-direct", "gemini-direct"


def _get_annotation(annotations: list, name: str) -> float | None:
    for a in annotations:
        if a.get("name") == name:
            return a.get("score")
    return None


def process_file(path: Path, cache: dict) -> list[dict]:
    system = _system_name(path)
    data = json.loads(path.read_text())
    rows = []
    total = len(data)

    for i, entry in enumerate(data):
        if entry.get("error"):
            continue

        dish_id = entry["input"]["dish_id"]
        rep = entry["repetition_number"]
        output = entry.get("output") or {}
        ref = entry.get("reference_output") or {}

        pred_ingredients = _parse_ingredients(output.get("ingredients"))
        gt_ingredients = _parse_ingredients(ref.get("ingredients"))
        predicted_calories = output.get("predicted_calories", 0)
        annotations = entry.get("annotations") or []

        calorie_mae = _get_annotation(annotations, "calorie_mae")
        calorie_pct_error = _get_annotation(annotations, "calorie_pct_error")

        # LLM judge
        key = _cache_key(pred_ingredients, gt_ingredients)
        if key not in cache:
            print(f"  [{i+1}/{total}] judging {system} / {dish_id} rep {rep}...")
            try:
                judgment = _judge_sync(pred_ingredients, gt_ingredients)
                cache[key] = [m.model_dump() for m in judgment.matches]
            except Exception as e:
                print(f"    WARNING: judge failed for {dish_id} rep {rep}: {e}")
                cache[key] = []
            _save_cache(cache)

        matches = cache[key]
        if not matches:
            recall = 1.0
            fuzzy = 0
        else:
            recall = sum(1 for m in matches if m["matched"]) / len(matches)
            fuzzy = sum(1 for m in matches if m["matched"] and not m["exact"])

        rows.append({
            "system": system,
            "dish_id": dish_id,
            "repetition": rep,
            "predicted_calories": predicted_calories,
            "calorie_mae": calorie_mae,
            "calorie_pct_error": calorie_pct_error,
            "ingredient_recall": round(recall, 4),
            "ingredient_fuzzy_matches": fuzzy,
        })

    return rows


def main(paths: list[Path]) -> None:
    import pandas as pd

    cache = _load_cache()
    all_rows = []

    for path in paths:
        print(f"\nProcessing {path.name}...")
        rows = process_file(path, cache)
        all_rows.extend(rows)
        print(f"  {len(rows)} rows processed")

    df = pd.DataFrame(all_rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {RESULTS_CSV}")
    print(f"Judge cache → {_CACHE_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.evaluate_ingredients <file.json> [file2.json ...]")
        sys.exit(1)
    files = [Path(p) for p in sys.argv[1:]]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"Files not found: {missing}")
        sys.exit(1)
    main(files)
