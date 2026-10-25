import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from backend.agents.vision import analyze_food_image
from backend.models import (
    AnalyzeResponse,
    AnalyzeResult,
    ConfirmRequest,
    ConfirmResponse,
    MealRecord,
)
from backend.services import fatsecret, db

app = FastAPI(title="Calorie Estimator")

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=FRONTEND_PATH.read_text())


class AnalyzeRequest(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    # 1. Vision agent: identify ingredients + estimated grams
    ingredient_list = await analyze_food_image(req.image_base64, req.media_type)

    if not ingredient_list.items:
        raise HTTPException(status_code=422, detail="No food detected in image")

    # 2. FatSecret lookup for each ingredient (parallel)
    async def lookup(ingredient):
        try:
            best, alternatives = await fatsecret.search_and_match(ingredient.name)
        except Exception:
            best, alternatives = None, []
        return AnalyzeResult(
            ingredient=ingredient,
            food_match=best,
            alternatives=alternatives,
        )

    results = await asyncio.gather(*[lookup(i) for i in ingredient_list.items])

    return AnalyzeResponse(results=list(results))


@app.post("/confirm", response_model=ConfirmResponse)
async def confirm(req: ConfirmRequest):
    if not req.ingredients:
        raise HTTPException(status_code=422, detail="No ingredients provided")

    total_calories = sum(
        (i.grams * i.calories_per_100g / 100.0) for i in req.ingredients
    )
    total_calories = round(total_calories, 1)

    ingredients_json = json.dumps([i.model_dump() for i in req.ingredients])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    rows = await db.query(
        "INSERT INTO meals (created_at, total_calories, ingredients_json) VALUES (?, ?, ?) RETURNING id",
        [now, total_calories, ingredients_json]
    )
    meal_id = rows[0]["id"] if rows else 0

    today_start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_rows = await db.query(
        "SELECT COALESCE(SUM(total_calories), 0) AS total FROM meals WHERE created_at >= ?",
        [today_start]
    )
    today_total = round(float(today_rows[0]["total"]) if today_rows else total_calories, 1)

    return ConfirmResponse(meal_id=meal_id, total_calories=total_calories, today_total=today_total)


@app.get("/meals", response_model=list[MealRecord])
async def list_meals():
    rows = await db.query(
        "SELECT id, created_at, total_calories, ingredients_json FROM meals ORDER BY created_at DESC LIMIT 10"
    )
    records = []
    for row in rows:
        try:
            ingredients = json.loads(row["ingredients_json"])
        except (json.JSONDecodeError, KeyError):
            ingredients = []
        records.append(MealRecord(
            id=row["id"],
            created_at=row["created_at"],
            total_calories=row["total_calories"],
            ingredients=ingredients,
        ))
    return records
