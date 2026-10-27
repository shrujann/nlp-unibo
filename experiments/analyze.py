"""
Post-hoc analysis of experiment results.

Run after experiments.run has completed and results.csv exists:
  python -m experiments.analyze

Produces:
  - Console summary table (mean ± std per system)
  - experiments/data/analysis.csv  (per-dish signed errors + derived columns)
  - experiments/data/figures/      (PNG plots)
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_CSV = Path(__file__).parent / "data" / "results.csv"
ANALYSIS_CSV = Path(__file__).parent / "data" / "analysis.csv"
FIGURES_DIR = Path(__file__).parent / "data" / "figures"
NUTRITION5K_CSV = Path(__file__).parent / "data" / "nutrition5k" / "sample.csv"

SYSTEMS = ["our-app", "gpt4o-mini-direct", "gemini-direct"]
SYSTEM_LABELS = {
    "our-app": "Our App (A)",
    "gpt4o-mini-direct": "GPT-4o-mini (B)",
    "gemini-direct": "Gemini Flash (C)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_gt_calories() -> dict[str, float]:
    """Return {dish_id: total_calories} from the Nutrition5k sample CSV."""
    df = pd.read_csv(NUTRITION5K_CSV)
    return dict(zip(df["dish_id"], df["total_calories"].astype(float)))


def _load_gt_ingredient_counts() -> dict[str, int]:
    """Return {dish_id: num_ingredients} from the Nutrition5k sample CSV."""
    df = pd.read_csv(NUTRITION5K_CSV)
    counts = {}
    for _, row in df.iterrows():
        n = sum(1 for i in range(1, 20) if pd.notna(row.get(f"ingr_{i}_name", None)) and str(row.get(f"ingr_{i}_name", "")).strip())
        counts[row["dish_id"]] = n
    return counts


def _fmt(mean: float, std: float, decimals: int = 1) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RESULTS_CSV.exists():
        print(f"No results found at {RESULTS_CSV} — run experiments.run first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)
    gt_calories = _load_gt_calories()
    gt_ingredient_counts = _load_gt_ingredient_counts()

    # ── Signed error (positive = overestimate, negative = underestimate) ───────
    df["gt_calories"] = df["dish_id"].map(gt_calories)
    df["signed_error"] = df["predicted_calories"] - df["gt_calories"]
    df["gt_ingredient_count"] = df["dish_id"].map(gt_ingredient_counts)

    df.to_csv(ANALYSIS_CSV, index=False)
    print(f"Per-dish analysis saved → {ANALYSIS_CSV}\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"{'System':<22} {'MAE (kcal)':>14} {'% Error':>12} {'Recall':>10} {'Fuzzy':>8} {'Signed Error':>14}")
    print("─" * 84)
    for system in SYSTEMS:
        s = df[df["system"] == system]
        per_rep = s.groupby("repetition").agg(
            mae=("calorie_mae", "mean"),
            pct=("calorie_pct_error", "mean"),
            recall=("ingredient_recall", "mean"),
            fuzzy=("ingredient_fuzzy_matches", "mean"),
            signed=("signed_error", "mean"),
        )
        print(
            f"{SYSTEM_LABELS[system]:<22}"
            f" {_fmt(per_rep['mae'].mean(), per_rep['mae'].std()):>14}"
            f" {_fmt(per_rep['pct'].mean(), per_rep['pct'].std()):>12}"
            f" {_fmt(per_rep['recall'].mean(), per_rep['recall'].std(), 3):>10}"
            f" {_fmt(per_rep['fuzzy'].mean(), per_rep['fuzzy'].std()):>8}"
            f" {_fmt(per_rep['signed'].mean(), per_rep['signed'].std()):>14}"
        )
    print()

    # ── Plot 1: Signed error distributions (box plot) ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df[df["system"] == s]["signed_error"].dropna().values for s in SYSTEMS]
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(SYSTEMS) + 1))
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEMS])
    ax.set_ylabel("Signed error (kcal)")
    ax.set_title("Calorie estimation bias per system\n(positive = overestimate, negative = underestimate)")
    plt.tight_layout()
    path = FIGURES_DIR / "signed_error_boxplot.png"
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)

    # ── Plot 2: MAE vs dish ingredient count ──────────────────────────────────
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(12, 4), sharey=True)
    for ax, system in zip(axes, SYSTEMS):
        s = df[df["system"] == system].dropna(subset=["calorie_mae", "gt_ingredient_count"])
        ax.scatter(s["gt_ingredient_count"], s["calorie_mae"], alpha=0.5, s=20, color="#4C72B0")
        m, b = np.polyfit(s["gt_ingredient_count"], s["calorie_mae"], 1)
        xs = np.linspace(s["gt_ingredient_count"].min(), s["gt_ingredient_count"].max(), 100)
        ax.plot(xs, m * xs + b, color="red", linewidth=1.2)
        ax.set_title(SYSTEM_LABELS[system])
        ax.set_xlabel("GT ingredient count")
    axes[0].set_ylabel("Calorie MAE (kcal)")
    fig.suptitle("MAE vs dish complexity", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "mae_vs_complexity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)

    # ── Plot 3: Recall vs MAE scatter ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for system, colour in zip(SYSTEMS, colours):
        s = df[df["system"] == system].dropna(subset=["ingredient_recall", "calorie_mae"])
        ax.scatter(s["ingredient_recall"], s["calorie_mae"], alpha=0.5, s=20,
                   color=colour, label=SYSTEM_LABELS[system])
    ax.set_xlabel("Ingredient recall")
    ax.set_ylabel("Calorie MAE (kcal)")
    ax.set_title("Recall vs calorie error")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "recall_vs_mae.png"
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
