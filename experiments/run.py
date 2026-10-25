"""
Run all three experiments against the persisted Nutrition5k-50 dataset in Phoenix.

Prerequisites:
  1. python -m experiments.download_dataset   (once — downloads images)
  2. python -m experiments.upload_dataset     (once — uploads to Phoenix)
  3. chainlit run app.py  (or any call to setup_phoenix()) so Phoenix is running

Then:
  python -m experiments.run

Results appear at http://localhost:6006 under the Experiments tab.
Each run creates a new experiment version so you can compare across runs.
"""
from dotenv import load_dotenv

load_dotenv()

from experiments.utils import ensure_phoenix

ensure_phoenix()

from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from experiments.evaluators import calorie_mae, calorie_pct_error, ingredient_recall
from experiments.tasks import gemini_direct_task, gpt4o_mini_task, our_app_task
from experiments.upload_dataset import DATASET_NAME

EVALUATORS = [calorie_mae, calorie_pct_error, ingredient_recall]

EXPERIMENTS = [
    (
        "our-app",
        our_app_task,
        "Gemini vision + FatSecret nutritional lookup",
    ),
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


def main() -> None:
    client = Client()
    dataset = client.datasets.get_dataset(dataset=DATASET_NAME)
    print(f"Running experiments on dataset '{DATASET_NAME}' ({len(dataset)} examples)\n")

    for name, task_fn, description in EXPERIMENTS:
        print(f"▶  {name}")
        run_experiment(
            dataset=dataset,
            task=task_fn,
            evaluators=EVALUATORS,
            experiment_name=name,
            experiment_description=description,
            client=client,
        )
        print(f"   done\n")

    print("All experiments complete — view results at http://localhost:6006")


if __name__ == "__main__":
    main()
