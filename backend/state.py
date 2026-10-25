from backend.models import AnalyzeResult, ConfirmIngredient

_CONF_EMOJI = {
    "high": "🟢",
    "mid": "🟡",
    "low": "🔴",
}


def _conf_label(confidence: float) -> str:
    if confidence >= 0.85:
        return f"{_CONF_EMOJI['high']} {int(confidence * 100)}%"
    elif confidence >= 0.6:
        return f"{_CONF_EMOJI['mid']} {int(confidence * 100)}%"
    else:
        return f"{_CONF_EMOJI['low']} {int(confidence * 100)}%"


def format_breakdown(results: list[AnalyzeResult], confirm_items: list[ConfirmIngredient]) -> str:
    rows = []
    total = 0.0
    for item in confirm_items:
        kcal = item.grams * item.calories_per_100g / 100.0
        total += kcal
        rows.append(f"| {item.name} | {item.grams:.0f}g | {kcal:.0f} kcal |")

    # Add confidence column if we have raw results for mapping
    conf_map = {r.ingredient.name.lower(): r.ingredient.confidence for r in results}

    header = f"**Meal breakdown — {total:.0f} kcal total**\n\n"
    header += "| Ingredient | Grams | Calories |\n"
    header += "|------------|-------|----------|\n"
    table = header + "\n".join(rows)
    table += f"\n\n*Type a tweak (e.g. \"remove the sauce\", \"change chicken to 100g\", \"add 1 egg\") or say **log this** to save.*"
    return table


def results_to_confirm(results: list[AnalyzeResult]) -> list[ConfirmIngredient]:
    items = []
    for r in results:
        if r.food_match:
            items.append(ConfirmIngredient(
                name=r.ingredient.name,
                grams=r.ingredient.estimated_grams,
                calories_per_100g=r.food_match.calories_per_100g,
                food_id=r.food_match.food_id,
            ))
        else:
            items.append(ConfirmIngredient(
                name=r.ingredient.name,
                grams=r.ingredient.estimated_grams,
                calories_per_100g=0.0,
            ))
    return items


def format_confirm_breakdown(items: list[ConfirmIngredient]) -> str:
    rows = []
    total = 0.0
    for item in items:
        kcal = item.grams * item.calories_per_100g / 100.0
        total += kcal
        rows.append(f"| {item.name} | {item.grams:.0f}g | {kcal:.0f} kcal |")

    header = f"**Meal breakdown — {total:.0f} kcal total**\n\n"
    header += "| Ingredient | Grams | Calories |\n"
    header += "|------------|-------|----------|\n"
    table = header + "\n".join(rows)
    table += f"\n\n*Type a tweak or say **log this** to save.*"
    return table
