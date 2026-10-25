import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from opentelemetry import trace
from typing_extensions import TypedDict

from backend.config import OPENROUTER_BASE_URL, VISION_MODEL
from backend.models import Ingredient, IngredientList

_tracer = trace.get_tracer("food_log.vision")

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a precise food portion analyst. When given a food image:

1. Identify the container/plate size first (standard plate ≈ 28 cm, bowl ≈ 500 ml, cup ≈ 250 ml)
2. List every distinct food component — main items, sauces, garnishes, condiments
3. Estimate each component's weight in grams using visual proportion and typical serving sizes
4. Assign confidence honestly (0.0–1.0):
   - 0.9–1.0: unambiguous (e.g. rice, broccoli, egg)
   - 0.6–0.8: likely but visually similar to alternatives (e.g. chicken vs fish, beef vs pork)
   - 0.3–0.5: uncertain — use notes to name alternatives (e.g. "could be salmon or chicken")
   - Never assign 1.0 to a cooked protein — texture and colour change during cooking"""

_SIMPLIFIED_SYSTEM = """You are a food analyst. Return a structured ingredient list.
Typical weights if unsure:
- Main protein (chicken, fish): 120–180g
- Rice/pasta: 150–200g
- Vegetables: 80–120g
- Sauce: 30–60g
- Bread slice: 30–40g"""


# ── Graph state ────────────────────────────────────────────────────────────────

class VisionState(TypedDict):
    image_base64: str
    media_type: str
    result: Optional[IngredientList]
    attempts: int
    use_simplified: bool


# ── Nodes ──────────────────────────────────────────────────────────────────────

def _make_llm(simplified: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=VISION_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=2048 if simplified else 4096,
    )


def analyze_node(state: VisionState) -> dict:
    simplified = state.get("use_simplified", False)
    llm = _make_llm(simplified).with_structured_output(IngredientList)

    messages = [
        SystemMessage(content=_SIMPLIFIED_SYSTEM if simplified else _SYSTEM),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{state['media_type']};base64,{state['image_base64']}"}
            },
            {"type": "text", "text": "Identify all ingredients and estimate their weights in grams."}
        ])
    ]

    try:
        result: IngredientList = llm.invoke(messages)
    except Exception:
        result = IngredientList(items=[])

    # Fill zero-gram items with 100g default on simplified retry
    if simplified and result.items:
        result = IngredientList(items=[
            Ingredient(
                name=i.name,
                estimated_grams=i.estimated_grams if i.estimated_grams > 0 else 100.0,
                confidence=i.confidence,
                notes=i.notes,
            )
            for i in result.items
        ])

    return {"result": result, "attempts": state["attempts"] + 1}


def prepare_retry(_state: VisionState) -> dict:
    return {"use_simplified": True}


# ── Routing ────────────────────────────────────────────────────────────────────

def should_retry(state: VisionState) -> str:
    result = state.get("result")
    if state["attempts"] >= 2:
        return END
    if not result or not result.items:
        return "prepare_retry"
    if all(i.estimated_grams == 0 for i in result.items):
        return "prepare_retry"
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────

_builder = StateGraph(VisionState)
_builder.add_node("analyze", analyze_node)
_builder.add_node("prepare_retry", prepare_retry)
_builder.set_entry_point("analyze")
_builder.add_conditional_edges("analyze", should_retry, {
    "prepare_retry": "prepare_retry",
    END: END,
})
_builder.add_edge("prepare_retry", "analyze")

_graph = _builder.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

async def analyze_food_image(image_base64: str, media_type: str = "image/jpeg") -> IngredientList:
    with _tracer.start_as_current_span("vision.analyze_food_image") as span:
        span.set_attribute("vision.model", VISION_MODEL)
        span.set_attribute("vision.media_type", media_type)

        final_state = await _graph.ainvoke({
            "image_base64": image_base64,
            "media_type": media_type,
            "result": None,
            "attempts": 0,
            "use_simplified": False,
        })

        result: IngredientList = final_state["result"] or IngredientList(items=[])
        attempts: int = final_state.get("attempts", 1)
        used_simplified: bool = final_state.get("use_simplified", False)

        span.set_attribute("vision.attempts", attempts)
        span.set_attribute("vision.used_simplified_prompt", used_simplified)
        span.set_attribute("vision.ingredient_count", len(result.items))

        for i, item in enumerate(result.items):
            span.set_attribute(f"vision.ingredient.{i}.name", item.name)
            span.set_attribute(f"vision.ingredient.{i}.estimated_grams", item.estimated_grams)
            span.set_attribute(f"vision.ingredient.{i}.confidence", item.confidence)
            if item.notes:
                span.set_attribute(f"vision.ingredient.{i}.notes", item.notes)

        return result
