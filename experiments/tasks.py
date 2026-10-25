"""
Task functions for each system under test.

Each function receives a Phoenix Example and returns:
  {"predicted_calories": int, "ingredients": [{"name", "grams", "calories"}, ...]}
"""
import base64
import json
import os

from dotenv import load_dotenv

load_dotenv()

from backend.agents.vision import analyze_food_image
from backend.config import OPENROUTER_BASE_URL, VISION_MODEL
from backend.services import fatsecret

_DIRECT_PROMPT = (
    "Estimate the total calories in this meal and list every ingredient with its weight.\n"
    "Return ONLY valid JSON — no markdown, no extra text:\n"
    '{"total_calories": <number>, "ingredients": [{"name": "...", "grams": <number>, "calories": <number>}]}'
)


def _encode(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), "image/png"


# ── System A: our app (Gemini vision → FatSecret lookup) ──────────────────────

async def our_app_task(example) -> dict:
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


# ── System B: GPT-4o direct (no FatSecret) ────────────────────────────────────

async def gpt4o_task(example) -> dict:
    from openai import AsyncOpenAI

    img_b64, mime = _encode(example.input["image_path"])
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": _DIRECT_PROMPT},
                ],
            }
        ],
        max_tokens=1024,
    )
    data = json.loads(response.choices[0].message.content)
    return {
        "predicted_calories": round(float(data.get("total_calories", 0))),
        "ingredients": data.get("ingredients", []),
    }


# ── System C: Gemini direct (no FatSecret) ────────────────────────────────────

async def gemini_direct_task(example) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    img_b64, mime = _encode(example.input["image_path"])
    llm = ChatOpenAI(
        model=VISION_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=1024,
    )
    msg = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": _DIRECT_PROMPT},
        ]
    )
    response = await llm.ainvoke([msg])

    text = response.content
    # Strip markdown fences if the model wraps the JSON
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
