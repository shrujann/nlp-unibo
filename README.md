# Retrieval-Augmented Calorie Estimation: An Agent Approach to Food Logging

A multimodal, conversational food-logging assistant that replaces self-reported calorie estimation with automated visual analysis grounded in a real nutritional database. Users photograph their meals, and the system identifies ingredients, estimates portion weights, and cross-references them against the FatSecret nutritional database to produce an objective calorie breakdown — which can be reviewed and corrected before logging.


---

## How it works

1. User uploads a meal photo in the chat interface
2. A vision LLM (Gemini 2.5 Flash Lite) identifies ingredients and estimates gram weights
3. Each ingredient is looked up in the FatSecret database in parallel; ambiguous matches trigger visual reranking
4. A formatted breakdown is returned for review — the user can edit, add, or remove ingredients
5. Once confirmed, the meal is logged to Cloudflare D1 and a calorie progress bar is updated inline

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A [Cloudflare](https://dash.cloudflare.com) account with D1 enabled
- API keys for [OpenRouter](https://openrouter.ai/keys) and [FatSecret](https://platform.fatsecret.com/api)

### Install dependencies

```bash
uv sync
```

### Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Where to get it |
|---|---|
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `FATSECRET_CONSUMER_KEY` / `SECRET` | [platform.fatsecret.com/api](https://platform.fatsecret.com/api) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare dashboard → right sidebar |
| `CF_D1_DATABASE_ID` | After creating the D1 database (see below) |
| `CLOUDFLARE_API_TOKEN` | Cloudflare → Profile → API Tokens (D1 Edit permission) |
| `CHAINLIT_AUTH_SECRET` | Run `chainlit create-secret` |

### Set up Cloudflare D1

```bash
npx wrangler d1 create nlp-food-log
npx wrangler d1 execute nlp-food-log --remote --file=migrations/001_init.sql
```

### Run the app

```bash
uv run chainlit run app.py
```

The app will be available at `http://localhost:8000`. Create an account on first launch — the onboarding flow will ask for your height, weight, age, activity level, and goal to compute your daily calorie target.

---

## Experiments

The evaluation compares three systems on a 50-dish subset of [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k):

| System | Model | Database |
|---|---|---|
| A — Our App | Gemini 2.5 Flash Lite | FatSecret |
| B — GPT-4o-mini direct | GPT-4o-mini | None |
| C — Gemini Flash direct | Gemini 2.5 Flash | None |

### Reproduce the results

**Step 1 — Download the dataset**

```bash
python -m experiments.download_dataset
```

This downloads 50 dish images and ground truth annotations into `experiments/data/nutrition5k/`.

**Step 2 — Run the experiments** (requires `PHOENIX_API_KEY` and `PHOENIX_BASE_URL` in `.env`)

```bash
python -m experiments.run
```

Runs all three systems with 5 repetitions each. Results are saved incrementally to `experiments/data/results.csv`. Experiment traces are stored in [Arize Phoenix](https://app.phoenix.arize.com).

**Step 3 — Run the ingredient judge**

```bash
python -m experiments.evaluate_ingredients \
  experiments/experiment-results/our-app.json \
  experiments/experiment-results/gpt4o-mini-direct.json \
  experiments/experiment-results/gemini-direct.json
```

Runs an LLM-as-a-judge evaluator on the ingredient lists from each system and writes `ingredient_recall` and `ingredient_fuzzy_matches` back into `results.csv`. Results are cached to `experiments/data/.judge_cache.json` so the script is resumable.

**Step 4 — Analyse**

```bash
python -m experiments.analyze
```

Prints a summary table (mean ± std across 5 repetitions for all metrics including signed error) and saves three figures to `experiments/data/figures/`.

### View the pre-run results

The raw Phoenix export files for all three systems are included in `experiments/experiment-results/`. To reproduce the analysis without re-running the experiments:

```bash
# Skip steps 1–2, start from step 3
python -m experiments.evaluate_ingredients \
  experiments/experiment-results/our-app.json \
  experiments/experiment-results/gpt4o-mini-direct.json \
  experiments/experiment-results/gemini-direct.json

python -m experiments.analyze
```

The summary table and figures you get will match the results reported in the paper.

---

## Project structure

```
app.py                          Chainlit entry point
backend/
  agents/
    vision.py                   Vision analysis agent (LangGraph)
    chat.py                     ReAct conversational agent (LangGraph)
  services/
    fatsecret.py                FatSecret API client + visual reranking
  config.py                     Model and API configuration
experiments/
  run.py                        Run all three systems via Arize Phoenix
  tasks.py                      Task functions for each system
  evaluators.py                 Calorie MAE, % error, LLM judge evaluators
  evaluate_ingredients.py       Post-hoc ingredient recall scoring
  analyze.py                    Summary table + figures
  experiment-results/           Raw Phoenix exports (our-app, gpt4o-mini, gemini)
  data/
    nutrition5k/                Dataset images and annotations
    results.csv                 Aggregated experiment results
    figures/                    Generated plots
migrations/
  001_init.sql                  Cloudflare D1 schema
```

---

## Stack

- **LLM inference** — [OpenRouter](https://openrouter.ai) (Gemini 2.5 Flash Lite, GPT-4o-mini, Gemini 2.5 Flash)
- **Agent framework** — [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://python.langchain.com)
- **Chat interface** — [Chainlit](https://chainlit.io)
- **Nutritional database** — [FatSecret API](https://platform.fatsecret.com/api)
- **Meal storage** — [Cloudflare D1](https://developers.cloudflare.com/d1/)
- **Experiment tracking** — [Arize Phoenix](https://app.phoenix.arize.com)
- **Dataset** — [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
