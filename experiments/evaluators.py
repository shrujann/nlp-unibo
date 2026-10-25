"""
Numeric evaluators for calorie estimation accuracy.

Each evaluator receives (output, expected) where:
  output   = what the task function returned
  expected = the dataset row's ground truth output fields
"""


def calorie_mae(output, expected) -> float:
    """Absolute error in kcal between prediction and ground truth."""
    predicted = (output or {}).get("predicted_calories", 0)
    ground_truth = (expected or {}).get("total_calories", 0)
    return abs(predicted - ground_truth)


def calorie_pct_error(output, expected) -> float:
    """Percentage error relative to ground truth total calories."""
    predicted = (output or {}).get("predicted_calories", 0)
    ground_truth = (expected or {}).get("total_calories", 0)
    if not ground_truth:
        return 0.0
    return abs(predicted - ground_truth) / ground_truth * 100


def ingredient_recall(output, expected) -> float:
    """
    Fraction of ground truth ingredients whose name appears (substring) in the
    predicted ingredient list.  0.0 = nothing matched, 1.0 = all matched.
    """
    predicted_names = {
        i["name"].lower() for i in (output or {}).get("ingredients", [])
    }
    gt_ingredients = (expected or {}).get("ingredients", [])
    if not gt_ingredients:
        return 1.0

    hits = sum(
        1
        for gt in gt_ingredients
        if any(
            gt["name"].lower() in p or p in gt["name"].lower()
            for p in predicted_names
        )
    )
    return hits / len(gt_ingredients)
