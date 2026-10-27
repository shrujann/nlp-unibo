"""
Numeric evaluators for calorie estimation accuracy.

Each evaluator receives (output, expected) where:
  output   = what the task function returned
  expected = the dataset row's ground truth output fields

ingredient_recall and ingredient_fuzzy_matches share a single LLM judge call
per dish via a module-level cache keyed on the ingredient lists.
"""
import asyncio
import hashlib
import json
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from backend.config import CHAT_MODEL, OPENROUTER_BASE_URL


# ── LLM judge ─────────────────────────────────────────────────────────────────

class _IngredientMatch(BaseModel):
    gt_name: str
    matched: bool
    exact: bool
    predicted_name: Optional[str] = None
    note: Optional[str] = None


class _IngredientJudgment(BaseModel):
    matches: list[_IngredientMatch]


_JUDGE_SYSTEM = """You are evaluating a food recognition system.
Given a ground truth ingredient list and a predicted ingredient list, determine for each ground truth ingredient whether the system correctly identified it.

Rules:
- matched=true if the predicted list contains the same ingredient (exact name, synonym, or same base ingredient with minor variation)
- exact=true only if the name is identical or a universally accepted synonym (e.g. "spring onion" / "scallion")
- exact=false when: same base ingredient but different cut/variety/preparation (e.g. "chicken breast" vs "chicken thigh"), different specificity (e.g. "chicken" vs "chicken breast"), or near-synonyms with a meaningful difference
- matched=false if the ingredient is genuinely absent from the predicted list
- For each fuzzy match (matched=true, exact=false), set note to briefly describe the discrepancy (e.g. "cut differs: predicted 'chicken thigh'", "less specific: predicted 'chicken'")
- Set predicted_name to the matched entry from the predicted list, or null if unmatched"""

_judgment_cache: dict[str, _IngredientJudgment] = {}


def _parse_ingredients(value) -> list:
    """Handle both list-of-dicts and JSON-serialised string (pandas uploads ingredients as a string)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value or []


def _cache_key(output: dict, expected: dict) -> str:
    blob = json.dumps({
        "pred": sorted(i["name"] for i in _parse_ingredients((output or {}).get("ingredients"))),
        "gt": sorted(i["name"] for i in _parse_ingredients((expected or {}).get("ingredients"))),
    }, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


async def _judge_async(output: dict, expected: dict) -> _IngredientJudgment:
    gt_ingredients = _parse_ingredients((expected or {}).get("ingredients"))
    predicted_ingredients = _parse_ingredients((output or {}).get("ingredients"))

    if not gt_ingredients:
        return _IngredientJudgment(matches=[])

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=1024,
        temperature=0,
    ).with_structured_output(_IngredientJudgment)

    gt_names = [i["name"] for i in gt_ingredients]
    pred_names = [i["name"] for i in predicted_ingredients]

    return await llm.ainvoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=f"Ground truth: {json.dumps(gt_names)}\nPredicted: {json.dumps(pred_names)}"),
    ])


def _get_judgment(output: dict, expected: dict) -> _IngredientJudgment:
    key = _cache_key(output, expected)
    if key not in _judgment_cache:
        _judgment_cache[key] = asyncio.run(_judge_async(output, expected))
    return _judgment_cache[key]


# ── Evaluators ─────────────────────────────────────────────────────────────────

def calorie_mae(output, expected) -> float:
    """Absolute error in kcal between prediction and ground truth."""
    predicted = float((output or {}).get("predicted_calories", 0))
    ground_truth = float((expected or {}).get("total_calories", 0))
    return abs(predicted - ground_truth)


def calorie_pct_error(output, expected) -> float:
    """Percentage error relative to ground truth total calories."""
    predicted = float((output or {}).get("predicted_calories", 0))
    ground_truth = float((expected or {}).get("total_calories", 0))
    if not ground_truth:
        return 0.0
    return abs(predicted - ground_truth) / ground_truth * 100


def ingredient_recall(output, expected) -> float:
    """Fraction of GT ingredients correctly identified (0–1), using an LLM judge for semantic matching."""
    judgment = _get_judgment(output, expected)
    if not judgment.matches:
        return 1.0
    return sum(1 for m in judgment.matches if m.matched) / len(judgment.matches)


def ingredient_fuzzy_matches(output, expected) -> int:
    """Count of GT ingredients matched semantically but not exactly (shares the LLM judge call with ingredient_recall)."""
    judgment = _get_judgment(output, expected)
    return sum(1 for m in judgment.matches if m.matched and not m.exact)
