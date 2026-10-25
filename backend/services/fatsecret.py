import base64
import hashlib
import hmac
import os
import secrets
import time
import urllib.parse
from typing import Optional

import httpx
from opentelemetry import trace

from backend.models import FoodMatch

_tracer = trace.get_tracer("food_log.fatsecret")

FATSECRET_API_URL = "https://platform.fatsecret.com/rest/server.api"
_TIMEOUT = 12.0
_AMBIGUITY_THRESHOLD = 0.10  # mark as ambiguous if top 2 cals/100g differ by less than 10%


# ── OAuth 1.0 signing ──────────────────────────────────────────────────────────

def _percent_encode(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _build_base_string(method: str, url: str, params: dict[str, str]) -> str:
    normalized = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}"
        for k, v in sorted(params.items())
    )
    return f"{method.upper()}&{_percent_encode(url)}&{_percent_encode(normalized)}"


def _sign(base_string: str, consumer_secret: str) -> str:
    signing_key = f"{_percent_encode(consumer_secret)}&"
    digest = hmac.new(
        signing_key.encode("utf-8"),
        base_string.encode("utf-8"),
        hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _signed_params(consumer_key: str, consumer_secret: str, api_params: dict[str, str]) -> dict[str, str]:
    oauth = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_version": "1.0",
    }
    combined = {**api_params, **oauth}
    base_string = _build_base_string("POST", FATSECRET_API_URL, combined)
    signature = _sign(base_string, consumer_secret)
    return {**combined, "oauth_signature": signature}


# ── Low-level HTTP ─────────────────────────────────────────────────────────────

async def _call(api_params: dict[str, str]) -> dict:
    consumer_key = os.environ["FATSECRET_CONSUMER_KEY"]
    consumer_secret = os.environ["FATSECRET_CONSUMER_SECRET"]
    signed = _signed_params(consumer_key, consumer_secret, api_params)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            FATSECRET_API_URL,
            data=signed,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return response.json()


# ── Response parsing ───────────────────────────────────────────────────────────

def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _parse_servings(food_raw: dict) -> list[dict]:
    servings_raw = food_raw.get("servings", {})
    if not isinstance(servings_raw, dict):
        return []
    return _as_list(servings_raw.get("serving"))


def _calories_per_100g(servings: list[dict]) -> Optional[float]:
    """Return kcal per 100g from a food's serving list."""
    # Prefer explicit 100g serving
    for s in servings:
        try:
            amount = float(s.get("metric_serving_amount") or 0)
            unit = str(s.get("metric_serving_unit") or "g").lower()
            if abs(amount - 100) < 0.5 and unit in ("g", "gram", "grams"):
                return float(s.get("calories") or 0)
        except (ValueError, TypeError):
            continue

    # Scale from any gram-based serving
    for s in servings:
        try:
            amount = float(s.get("metric_serving_amount") or 0)
            unit = str(s.get("metric_serving_unit") or "g").lower()
            if amount > 0 and unit in ("g", "gram", "grams"):
                calories = float(s.get("calories") or 0)
                return calories * (100.0 / amount)
        except (ValueError, TypeError):
            continue

    # Last resort: first serving, assume it's ~100g
    if servings:
        try:
            return float(servings[0].get("calories") or 0)
        except (ValueError, TypeError):
            pass
    return None


def _best_serving_description(servings: list[dict]) -> str:
    if not servings:
        return "unknown serving"
    # Prefer the 100g serving description
    for s in servings:
        try:
            amount = float(s.get("metric_serving_amount") or 0)
            unit = str(s.get("metric_serving_unit") or "g").lower()
            if abs(amount - 100) < 0.5 and unit in ("g", "gram", "grams"):
                return str(s.get("serving_description") or "100g")
        except (ValueError, TypeError):
            pass
    return str(servings[0].get("serving_description") or "1 serving")


# ── Public API ─────────────────────────────────────────────────────────────────

async def search_and_match(ingredient_name: str) -> tuple[Optional[FoodMatch], list[FoodMatch]]:
    """
    Search FatSecret for an ingredient. Returns (best_match, alternatives).
    If the top 2 results are close in caloric density, best_match.ambiguous = True.
    """
    with _tracer.start_as_current_span("fatsecret.search_and_match") as span:
        span.set_attribute("ingredient.query", ingredient_name)

        json_data = await _call({
            "method": "foods.search",
            "search_expression": ingredient_name,
            "format": "json",
            "max_results": "5",
            "page_number": "0",
        })

        if "error" in json_data:
            span.set_attribute("result.error", str(json_data["error"]))
            return None, []

        foods_container = json_data.get("foods", {})
        raw_foods = _as_list(foods_container.get("food")) if isinstance(foods_container, dict) else []
        span.set_attribute("result.raw_count", len(raw_foods))

        if not raw_foods:
            return None, []

        # Basic-tier search doesn't return serving data — need to fetch each food
        matches: list[FoodMatch] = []
        for raw in raw_foods[:3]:
            food_id = str(raw.get("food_id", ""))
            food_name = str(raw.get("food_name", ""))
            if not food_id:
                continue

            # Fetch full food record to get serving/calorie data
            full = await _get_food_detail(food_id)
            if not full:
                continue

            cal = _calories_per_100g(_parse_servings(full))
            if cal is None or cal <= 0:
                continue

            matches.append(FoodMatch(
                food_id=food_id,
                food_name=food_name,
                calories_per_100g=round(cal, 1),
                matched_serving=_best_serving_description(_parse_servings(full)),
                food_image_url=_parse_food_image_url(full),
                allergens=_parse_allergens(full),
            ))

            if len(matches) >= 3:
                break

        if not matches:
            span.set_attribute("result.matched", 0)
            return None, []

        best = matches[0]
        alternatives = matches[1:]
        ambiguous = False

        # Mark as ambiguous when top 2 are within the threshold
        if alternatives:
            second = alternatives[0]
            diff = abs(best.calories_per_100g - second.calories_per_100g)
            ratio = diff / max(best.calories_per_100g, 1)
            if ratio < _AMBIGUITY_THRESHOLD:
                best = FoodMatch(**{**best.model_dump(), "ambiguous": True})
                ambiguous = True

        span.set_attribute("result.matched", len(matches))
        span.set_attribute("result.best.food_id", best.food_id)
        span.set_attribute("result.best.food_name", best.food_name)
        span.set_attribute("result.best.calories_per_100g", best.calories_per_100g)
        span.set_attribute("result.ambiguous", ambiguous)
        span.set_attribute("result.alternatives", [m.food_name for m in alternatives])

        return best, alternatives


def _parse_food_image_url(food_raw: dict) -> Optional[str]:
    images = food_raw.get("food_images")
    if not images:
        return None
    image_list = _as_list(images.get("food_image"))
    # Prefer "front" type
    for img in image_list:
        if isinstance(img, dict) and str(img.get("image_type", "")).lower() == "front":
            return img.get("image_url")
    first = image_list[0] if image_list else None
    return first.get("image_url") if isinstance(first, dict) else None


def _parse_allergens(food_raw: dict) -> list[str]:
    attrs = food_raw.get("food_attributes") or {}
    allergen_container = attrs.get("allergens") or {}
    allergens = _as_list(allergen_container.get("allergen"))
    return [
        str(a["allergen_name"])
        for a in allergens
        if isinstance(a, dict) and str(a.get("allergen_value", "0")) == "1"
    ]


async def _get_food_detail(food_id: str) -> Optional[dict]:
    json_data = await _call({
        "method": "food.get",
        "food_id": food_id,
        "format": "json",
        "include_food_images": "1",
        "include_food_attributes": "1",
    })
    if "error" in json_data:
        return None
    return json_data.get("food")
