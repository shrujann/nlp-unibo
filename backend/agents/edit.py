import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from backend.config import OPENROUTER_BASE_URL, CHAT_MODEL
from backend.models import ConfirmIngredient

_SYSTEM = """You are a meal editor. The user has a meal breakdown and wants to modify it.

Given:
- The current ingredient list (name, grams, calories_per_100g)
- The user's edit instruction

Apply the edit and return the full updated ingredient list. Rules:
- "remove X" / "no X" → delete that ingredient
- "add X" / "add Xg of Y" → add a new ingredient (use a reasonable calories_per_100g estimate if unknown)
- "change X to Yg" / "X should be Yg" → update grams for that ingredient
- "replace X with Y" → rename and update calories estimate
- Keep all other ingredients unchanged
- Return ALL ingredients (not just changed ones)
- Estimate calories_per_100g from common knowledge if adding a new item (e.g. egg ≈ 155, white rice ≈ 130, chicken breast ≈ 165, broccoli ≈ 35)"""


class EditedList(BaseModel):
    items: list[ConfirmIngredient]


async def edit_meal(current: list[ConfirmIngredient], instruction: str) -> list[ConfirmIngredient]:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=1024,
    ).with_structured_output(EditedList)

    current_text = json.dumps([i.model_dump() for i in current], indent=2)

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Current ingredients:\n{current_text}\n\nEdit instruction: {instruction}"),
    ]

    try:
        result: EditedList = await llm.ainvoke(messages)
        return result.items
    except Exception:
        return current
