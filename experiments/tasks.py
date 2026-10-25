"""
Task functions for each system under test.

Each function receives a Phoenix Example and returns:
  {"predicted_calories": int, "ingredients": [{"name", "grams", "calories"}, ...]}

All three systems use OpenRouter (OPENROUTER_API_KEY) — no separate API keys needed.

Models used:
  A  our-app       google/gemini-2.5-flash-lite  + FatSecret nutritional DB
  B  gpt4o-mini    openai/gpt-4o-mini            direct estimation (~$0.165/M input)
  C  gemini-flash  google/gemini-2.5-flash        direct estimation (~$0.30/M input)
"""
import asyncio
import base64
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

from backend.agents.vision import analyze_food_image
from backend.config import OPENROUTER_BASE_URL
from backend.services import fatsecret

# For the "direct" systems — same prompt, expect raw JSON back
_DIRECT_PROMPT = (
    "Estimate the total calories in this meal and list every ingredient with its weight.\n"
    "Return ONLY valid JSON — no markdown, no extra text:\n"
    '{"total_calories": <number>, "ingredients": [{"name": "...", "grams": <number>, "calories": <number>}]}'
)

# Cheap OpenAI vision model via OpenRouter
_GPT4O_MINI = "openai/gpt-4o-mini"

# Full Gemini Flash (not lite) — same family as the app, different tier
_GEMINI_FLASH = "google/gemini-2.5-flash"


def _encode(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), "image/png"


def _direct_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=1024,
    )


def _parse_direct_response(text: str) -> dict:
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    try:
        data = json.loads(text)
    except Exception:
        return {"predicted_calories": 0, "ingredients": []}
    return {
        "predicted_calories": round(float(data.get("total_calories", 0))),
        "ingredients": data.get("ingredients", []),
    }


# ── System A: our app (Gemini vision → FatSecret lookup) ──────────────────────

async def _our_app_task_async(example) -> dict:
    img_b64, mime = _encode(example.input["image_path"])
    ingredient_list = await analyze_food_image(img_b64, mime)

    total = 0.0
    details = []
    for ingr in ingredient_list.items:
        try:
            best, _ = await fatsecret.search_and_match(ingr.name)
        except Exception:
            best = None
        if best:
            kcal = ingr.estimated_grams * best.calories_per_100g / 100.0
            total += kcal
            details.append(
                {
                    "name": ingr.name,
                    "grams": round(ingr.estimated_grams, 1),
                    "calories": round(kcal, 1),
                }
            )

    return {"predicted_calories": round(total), "ingredients": details}


def our_app_task(example) -> dict:
    return asyncio.run(_our_app_task_async(example))


# ── System B: GPT-4o-mini direct via OpenRouter (no FatSecret) ────────────────

async def _gpt4o_mini_task_async(example) -> dict:
    img_b64, mime = _encode(example.input["image_path"])
    llm = _direct_llm(_GPT4O_MINI)
    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": _DIRECT_PROMPT},
        ]
    )
    response = await llm.ainvoke([msg])
    return _parse_direct_response(response.content)


def gpt4o_mini_task(example) -> dict:
    return asyncio.run(_gpt4o_mini_task_async(example))


# ── System C: Gemini 2.5 Flash direct via OpenRouter (no FatSecret) ───────────

async def _gemini_direct_task_async(example) -> dict:
    img_b64, mime = _encode(example.input["image_path"])
    llm = _direct_llm(_GEMINI_FLASH)
    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": _DIRECT_PROMPT},
        ]
    )
    response = await llm.ainvoke([msg])
    return _parse_direct_response(response.content)


def gemini_direct_task(example) -> dict:
    return asyncio.run(_gemini_direct_task_async(example))
