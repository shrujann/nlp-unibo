from pydantic import BaseModel
from typing import Optional


class Ingredient(BaseModel):
    name: str
    estimated_grams: float
    confidence: float
    notes: Optional[str] = None


class IngredientList(BaseModel):
    items: list[Ingredient]


class FoodMatch(BaseModel):
    food_id: str
    food_name: str
    calories_per_100g: float
    matched_serving: str
    ambiguous: bool = False
    food_image_url: Optional[str] = None
    allergens: list[str] = []


class AnalyzeResult(BaseModel):
    ingredient: Ingredient
    food_match: Optional[FoodMatch] = None
    alternatives: list[FoodMatch] = []


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResult]


class ConfirmIngredient(BaseModel):
    name: str
    grams: float
    calories_per_100g: float
    food_id: Optional[str] = None


class ConfirmRequest(BaseModel):
    ingredients: list[ConfirmIngredient]


class ConfirmResponse(BaseModel):
    meal_id: int
    total_calories: float
    today_total: float


class MealRecord(BaseModel):
    id: int
    created_at: str
    total_calories: float
    ingredients: list[ConfirmIngredient]
