"""
Run all three experiments against the Nutrition5k-50 dataset in Arize Phoenix cloud.

Prerequisites:
  1. python -m experiments.download_dataset   (once — downloads images)
  2. Add PHOENIX_API_KEY to your .env (from app.phoenix.arize.com)

Then:
  python -m experiments.run

Each system runs with repetitions=5 (Phoenix handles the 5 trials natively).
Results are saved to experiments/data/results.csv after each system completes,
so a crash mid-run doesn't lose completed work.

View results at https://app.phoenix.arize.com
"""
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from experiments.evaluators import calorie_mae, calorie_pct_error
from experiments.tasks import gemini_direct_task, gpt4o_mini_task, our_app_task  # noqa: F401
from experiments.upload_dataset import DATASET_NAME, upload

EVALUATORS = [calorie_mae, calorie_pct_error]
EVALUATOR_NAMES = ["calorie_mae", "calorie_pct_error"]
REPETITIONS = 5

EXPERIMENTS = [
    # (
    #     "our-app",
    #     our_app_task,
    #     "Gemini vision + FatSecret nutritional lookup",
    # ),
    (
        "gpt4o-mini-direct",
        gpt4o_mini_task,
        "GPT-4o-mini direct calorie estimation via OpenRouter (no nutritional DB)",
    ),
    (
        "gemini-direct",
        gemini_direct_task,
        "Gemini direct calorie estimation (no nutritional DB)",
    ),
]

RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_CSV = RESULTS_DIR / "results.csv"
FIELDNAMES = ["repetition", "system", "dish_id", "predicted_calories", "ingredients"] + EVALUATOR_NAMES


def _make_client() -> Client:
    return Client(
        base_url=os.environ["PHOENIX_BASE_URL"],
        api_key=os.environ["PHOENIX_API_KEY"],
    )


def _append_csv(system_name: str, ran: object) -> None:
    """Append rows for one completed system to the results CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    eval_index: dict[tuple[str, str], float] = {}
    for ev_run in ran["evaluation_runs"]:
        result = ev_run["result"] if isinstance(ev_run, dict) else getattr(ev_run, "result", None)
        if result is None:
            continue
        evals = result if isinstance(result, list) else [result]
        for ev in evals:
            score = ev.get("score") if isinstance(ev, dict) else getattr(ev, "score", None)
            if score is not None:
                run_id_key = ev_run["experiment_run_id"] if isinstance(ev_run, dict) else ev_run.experiment_run_id
                name_key = ev_run["name"] if isinstance(ev_run, dict) else ev_run.name
                eval_index[(run_id_key, name_key)] = score

    rows = []
    for task_run in ran["task_runs"]:
        run_id = task_run["id"] if isinstance(task_run, dict) else task_run.id
        output = (task_run.get("output") if isinstance(task_run, dict) else getattr(task_run, "output", None)) or {}
        rep = (task_run.get("repetition_number") if isinstance(task_run, dict) else getattr(task_run, "repetition_number", ""))
        ingredients = output.get("ingredients", []) if isinstance(output, dict) else []
        row = {
            "repetition": rep,
            "system": system_name,
            "dish_id": output.get("dish_id", "") if isinstance(output, dict) else "",
            "predicted_calories": output.get("predicted_calories", "") if isinstance(output, dict) else "",
            "ingredients": json.dumps(ingredients),
        }
        for ev_name in EVALUATOR_NAMES:
            row[ev_name] = eval_index.get((run_id, ev_name), "")
        rows.append(row)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    client = _make_client()
    upload()  # no-op if dataset already exists
    dataset = client.datasets.get_dataset(dataset=DATASET_NAME)
    print(f"Dataset '{DATASET_NAME}' — {len(dataset)} examples × {REPETITIONS} repetitions\n")

    for name, task_fn, description in EXPERIMENTS:
        print(f"▶  {name}")
        ran = run_experiment(
            dataset=dataset,
            task=task_fn,
            evaluators=EVALUATORS,
            experiment_name=name,
            experiment_description=description,
            repetitions=REPETITIONS,
            client=client,
        )
        _append_csv(name, ran)
        print(f"   done — results saved to {RESULTS_CSV}\n")

    print(f"All experiments complete.\nResults → {RESULTS_CSV}\nPhoenix → https://app.phoenix.arize.com")


if __name__ == "__main__":
    main()
