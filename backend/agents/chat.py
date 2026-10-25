import asyncio
import base64
import json
import os
from datetime import datetime, timedelta, timezone

import chainlit as cl
import httpx
from langchain_core.messages import AIMessageChunk, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from opentelemetry import trace
from pydantic import BaseModel

from backend.agents.vision import analyze_food_image
from backend.config import CHAT_MODEL, OPENROUTER_BASE_URL, VISION_MODEL
from backend.models import AnalyzeResult, ConfirmIngredient, FoodMatch
from backend.services import db, fatsecret
from backend.state import format_confirm_breakdown, results_to_confirm

_tracer = trace.get_tracer("food_log.rerank")


class _EditedList(BaseModel):
    items: list[ConfirmIngredient]


_EDIT_SYSTEM = """You are a meal editor. Apply the edit instruction to the ingredient list and return the full updated list.
Rules:
- "remove X" / "no X" → delete that ingredient
- "add X" / "add Xg of Y" → add new ingredient with a calorie estimate
- "change X to Yg" / "X should be Yg" → update grams for that ingredient
- Keep all other ingredients unchanged and return ALL of them
Common kcal/100g estimates: egg=155, white rice=130, chicken breast=165, broccoli=35, salmon=208, pasta=131, olive oil=884, butter=717"""


async def _visual_rerank(
    ingredient_name: str,
    user_b64: str,
    user_mime: str,
    candidates: list[FoodMatch],
) -> list[FoodMatch]:
    """Download FatSecret candidate images and ask Gemini which best matches the ingredient."""
    with _tracer.start_as_current_span("rerank.visual_rerank") as span:
        span.set_attribute("rerank.ingredient", ingredient_name)
        span.set_attribute("rerank.candidates", [c.food_name for c in candidates])

        with_images = [c for c in candidates if c.food_image_url]
        if len(with_images) < 2:
            span.set_attribute("rerank.skipped_reason", "fewer than 2 candidates have images")
            return candidates

        fetched: list[tuple[FoodMatch, str, str]] = []
        async with httpx.AsyncClient(timeout=6.0) as client:
            for candidate in with_images[:3]:
                try:
                    resp = await client.get(candidate.food_image_url)
                    if resp.status_code == 200:
                        img_b64 = base64.b64encode(resp.content).decode()
                        img_mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                        fetched.append((candidate, img_b64, img_mime))
                except Exception:
                    pass

        if len(fetched) < 2:
            span.set_attribute("rerank.skipped_reason", "failed to fetch enough candidate images")
            return candidates

        span.set_attribute("rerank.images_fetched", len(fetched))

        content = [
            {
                "type": "text",
                "text": (
                    f"Image 1 is a meal photo. Images 2–{len(fetched)+1} are food database photos numbered 1–{len(fetched)}. "
                    f"Which database number (1–{len(fetched)}) best matches '{ingredient_name}' from the meal photo? "
                    f"Reply with only the digit."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:{user_mime};base64,{user_b64}"}},
        ]
        for _, img_b64, img_mime in fetched:
            content.append({"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}})

        llm = ChatOpenAI(
            model=VISION_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            max_tokens=5,
        )
        try:
            response = await llm.ainvoke([{"role": "user", "content": content}])
            idx = int(response.content.strip()[0]) - 1
            if 0 <= idx < len(fetched):
                best = fetched[idx][0]
                span.set_attribute("rerank.selected_index", idx)
                span.set_attribute("rerank.selected_food_name", best.food_name)
                span.set_attribute("rerank.selected_calories_per_100g", best.calories_per_100g)
                reranked = [best] + [c for c, _, _ in fetched if c.food_id != best.food_id]
                without_images = [c for c in candidates if not c.food_image_url]
                return reranked + without_images
        except Exception:
            span.set_attribute("rerank.skipped_reason", "model response unparseable")

        return candidates


def make_tools(user_id: str) -> list:

    @tool
    async def analyze_food() -> str:
        """Analyse the uploaded food photo and return a calorie breakdown. Always call this when the user uploads an image."""
        pending = cl.user_session.get("pending_image")
        if not pending:
            return "No image found. Ask the user to upload a food photo."

        cl.user_session.set("pending_image", None)
        ingredient_list = await analyze_food_image(pending["b64"], pending["mime"])

        if not ingredient_list.items:
            return "Could not detect food in the image. Ask the user to try a clearer photo."

        async def lookup(ingredient):
            try:
                best, alts = await fatsecret.search_and_match(ingredient.name)
            except Exception:
                best, alts = None, []
            return AnalyzeResult(ingredient=ingredient, food_match=best, alternatives=alts)

        results = list(await asyncio.gather(*[lookup(i) for i in ingredient_list.items]))

        # Visual reranking: for ambiguous matches, use Gemini to pick the best candidate image
        reranked_results = []
        for result in results:
            if result.food_match and result.food_match.ambiguous and result.alternatives:
                all_candidates = [result.food_match] + result.alternatives
                reranked = await _visual_rerank(
                    result.ingredient.name,
                    pending["b64"],
                    pending["mime"],
                    all_candidates,
                )
                best = FoodMatch(**{**reranked[0].model_dump(), "ambiguous": False})
                reranked_results.append(AnalyzeResult(
                    ingredient=result.ingredient,
                    food_match=best,
                    alternatives=reranked[1:],
                ))
            else:
                reranked_results.append(result)

        confirm_items = results_to_confirm(reranked_results)
        cl.user_session.set("meal", confirm_items)

        # Collect allergens across all matched foods
        allergens: set[str] = set()
        for r in reranked_results:
            if r.food_match:
                allergens.update(r.food_match.allergens)

        breakdown = format_confirm_breakdown(confirm_items)
        if allergens:
            breakdown += f"\n\n⚠️ Allergens: {', '.join(sorted(allergens))}"
        return breakdown

    @tool
    async def edit_meal(instruction: str) -> str:
        """Apply a natural language edit to the current meal breakdown. Use when the user says things like 'remove the sauce', 'change chicken to 100g', 'add an egg'."""
        current: list[ConfirmIngredient] | None = cl.user_session.get("meal")
        if not current:
            return "No meal is loaded. Ask the user to upload a photo first."

        llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            max_tokens=1024,
        ).with_structured_output(_EditedList)

        current_text = json.dumps([i.model_dump() for i in current], indent=2)
        result: _EditedList = await llm.ainvoke([
            SystemMessage(content=_EDIT_SYSTEM),
            HumanMessage(content=f"Current ingredients:\n{current_text}\n\nEdit: {instruction}"),
        ])
        cl.user_session.set("meal", result.items)
        return format_confirm_breakdown(result.items)

    @tool
    async def log_meal() -> str:
        """Save (or update) the current meal breakdown. Call when the user says 'log this', 'save', 'update', 'done', or wants to record/overwrite the meal."""
        meal: list[ConfirmIngredient] | None = cl.user_session.get("meal")
        if not meal:
            return "No meal is loaded. Ask the user to upload a photo first."

        total = round(sum(i.grams * i.calories_per_100g / 100.0 for i in meal), 1)
        ingredients_json = json.dumps([i.model_dump() for i in meal])
        editing_id = cl.user_session.get("editing_meal_id")

        if editing_id:
            await db.query(
                "UPDATE meals SET total_calories=?, ingredients_json=? WHERE id=? AND user_id=?",
                [total, ingredients_json, editing_id, user_id],
            )
            cl.user_session.set("editing_meal_id", None)
            verb = "updated"
        else:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            await db.query(
                "INSERT INTO meals (created_at, total_calories, ingredients_json, user_id) VALUES (?,?,?,?)",
                [now, total, ingredients_json, user_id],
            )
            verb = "logged"

        cl.user_session.set("meal", None)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = await db.query(
            "SELECT COALESCE(SUM(total_calories),0) AS t, COUNT(*) AS c FROM meals WHERE user_id=? AND created_at>=?",
            [user_id, today],
        )
        today_total = round(float(rows[0]["t"])) if rows else total
        cnt = int(rows[0]["c"]) if rows else 1
        u_rows = await db.query("SELECT daily_target FROM users WHERE id=?", [user_id])
        target = int(u_rows[0]["daily_target"]) if u_rows and u_rows[0].get("daily_target") else None
        pct = min(today_total / target, 1.0) if target else None
        bar = f"  {'█' * round(pct * 10)}{'░' * (10 - round(pct * 10))} {int(pct*100)}%" if pct is not None else ""
        target_str = f" / {target:,}" if target else ""
        return f"Meal {verb} ({total:.0f} kcal). Today: {today_total:,.0f}{target_str} kcal{bar}"

    @tool
    async def get_today_summary() -> str:
        """Get the user's calorie total for today and remaining budget. Call when asked about current total, progress, or remaining calories."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = await db.query(
            "SELECT COALESCE(SUM(total_calories),0) AS t, COUNT(*) AS c FROM meals WHERE user_id=? AND created_at>=?",
            [user_id, today],
        )
        kcal = round(float(rows[0]["t"])) if rows else 0
        cnt = int(rows[0]["c"]) if rows else 0

        u_rows = await db.query("SELECT daily_target, goal FROM users WHERE id=?", [user_id])
        target = int(u_rows[0]["daily_target"]) if u_rows and u_rows[0].get("daily_target") else None

        if target:
            pct = min(kcal / target, 1.0)
            bar = "█" * round(pct * 10) + "░" * (10 - round(pct * 10))
            remaining = max(target - kcal, 0)
            return (
                f"Today: {kcal:,} / {target:,} kcal  {bar} {int(pct*100)}%\n"
                f"{cnt} meal{'s' if cnt != 1 else ''} logged · {remaining:,} kcal remaining"
            )
        return f"Today: {kcal:,} kcal across {cnt} meal{'s' if cnt != 1 else ''}."

    @tool
    async def get_meal_history(days: int = 7) -> str:
        """Get the user's meal history for the past N days. Call when asked about past meals, what they ate, or a weekly summary. Meals are numbered 1 (most recent) onwards."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = await db.query(
            "SELECT created_at, total_calories, ingredients_json FROM meals WHERE user_id=? AND created_at>=? ORDER BY created_at DESC",
            [user_id, since],
        )
        if not rows:
            return f"No meals logged in the past {days} day{'s' if days != 1 else ''}."

        lines = []
        for i, row in enumerate(rows, 1):
            dt = row["created_at"][:16]
            kcal = round(float(row["total_calories"]))
            try:
                items = json.loads(row["ingredients_json"])
                names = ", ".join(it["name"] for it in items[:3])
                if len(items) > 3:
                    names += f" +{len(items)-3} more"
            except Exception:
                names = "unknown"
            lines.append(f"{i}. {dt} — {kcal:,} kcal ({names})")

        total = sum(round(float(r["total_calories"])) for r in rows)
        avg = round(total / max(days, 1))
        return "\n".join(lines) + f"\n\nTotal: {total:,} kcal over {days} days · avg {avg:,}/day"

    async def _resolve_meal(position: int) -> tuple[int, dict] | tuple[None, None]:
        """Return (real_db_id, row) for the Nth most recent meal (1-indexed). Internal helper."""
        rows = await db.query(
            "SELECT id, created_at, total_calories, ingredients_json FROM meals WHERE user_id=? ORDER BY created_at DESC LIMIT 1 OFFSET ?",
            [user_id, position - 1],
        )
        if not rows:
            return None, None
        return rows[0]["id"], rows[0]

    @tool
    async def delete_meal(position: int) -> str:
        """Delete a previously logged meal by its position number (1 = most recent). Call when the user asks to delete or undo a meal. Use get_meal_history first to confirm the position."""
        real_id, row = await _resolve_meal(position)
        if real_id is None:
            return f"Meal {position} not found."

        kcal = round(float(row["total_calories"]))
        await db.query("DELETE FROM meals WHERE id=? AND user_id=?", [real_id, user_id])

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary = await db.query(
            "SELECT COALESCE(SUM(total_calories),0) AS t, COUNT(*) AS c FROM meals WHERE user_id=? AND created_at>=?",
            [user_id, today],
        )
        today_total = round(float(summary[0]["t"])) if summary else 0
        cnt = int(summary[0]["c"]) if summary else 0
        u_rows = await db.query("SELECT daily_target FROM users WHERE id=?", [user_id])
        target = int(u_rows[0]["daily_target"]) if u_rows and u_rows[0].get("daily_target") else None
        pct = min(today_total / target, 1.0) if target else None
        bar = f"  {'█' * round(pct * 10)}{'░' * (10 - round(pct * 10))} {int(pct*100)}%" if pct is not None else ""
        target_str = f" / {target:,}" if target else ""
        return f"Deleted meal {position} ({kcal} kcal). Today: {today_total:,}{target_str} kcal{bar} across {cnt} meal{'s' if cnt != 1 else ''}."

    @tool
    async def load_saved_meal(position: int) -> str:
        """Load a previously logged meal back into the editing session by its position (1 = most recent). Call when the user wants to edit a past meal. Use get_meal_history first to confirm the position."""
        real_id, row = await _resolve_meal(position)
        if real_id is None:
            return f"Meal {position} not found."

        try:
            items = [ConfirmIngredient(**i) for i in json.loads(row["ingredients_json"])]
        except Exception:
            return "Could not load meal — the data may be malformed."

        cl.user_session.set("meal", items)
        cl.user_session.set("editing_meal_id", real_id)
        dt = row["created_at"][:16]
        return f"Loaded meal {position} from {dt} for editing.\n\n" + format_confirm_breakdown(items) + "\n\nType an edit or say **update** to save changes."

    @tool
    async def get_user_profile() -> str:
        """Get the user's profile: TDEE, daily calorie target, goal, and body stats. Call when asked about their target, goal, or profile."""
        rows = await db.query("SELECT * FROM users WHERE id=?", [user_id])
        if not rows:
            return "Profile not found."
        u = rows[0]
        goal_map = {"lose": "Lose weight (−500 kcal/day)", "maintain": "Maintain", "gain": "Gain weight (+300 kcal/day)"}
        act_map = {"sedentary": "Sedentary", "light": "Lightly active", "moderate": "Moderately active", "active": "Active", "very_active": "Very active"}
        target = u.get("daily_target")
        tdee = u.get("tdee")
        return (
            f"**{user_id}'s profile**\n"
            f"- {u.get('sex','—')} · Age {u.get('age','—')} · {u.get('height_cm','—')} cm · {u.get('weight_kg','—')} kg\n"
            f"- Activity: {act_map.get(u.get('activity',''), '—')}\n"
            f"- Goal: {goal_map.get(u.get('goal',''), '—')}\n"
            f"- TDEE: {tdee:,} kcal · **Daily target: {target:,} kcal**" if target and tdee
            else "- Profile incomplete — run onboarding to set a target."
        )

    return [analyze_food, edit_meal, log_meal, get_today_summary, get_meal_history, get_user_profile, delete_meal, load_saved_meal]


def _system_prompt(username: str, daily_target: int | None, goal: str | None) -> str:
    target_line = f"- Daily calorie target: {daily_target:,} kcal" if daily_target else "- Daily calorie target: not set"
    goal_line = f"- Goal: {goal}" if goal else ""
    return f"""You are a personal food logging assistant for {username}.

Tools available:
- analyze_food: run when the user uploads a photo
- edit_meal: apply a natural language change to the current breakdown
- log_meal: save or update the current breakdown (also called when user says 'update' after editing a saved meal)
- get_today_summary: fetch today's calorie total from the database
- get_meal_history: fetch past meals with their IDs
- get_user_profile: show the user's TDEE, target, and stats
- delete_meal: permanently delete a logged meal by position number (1=most recent); call get_meal_history first
- load_saved_meal: load a past meal back into the editor by position number; call get_meal_history first

Meals are always referenced by position (1, 2, 3…), never by database ID. Never mention internal IDs to the user.

{target_line}
{goal_line}

## Tone and style — follow these strictly

You text like a knowledgeable friend, not a fitness app. Practical rules:

- **No filler openers.** Never start with "Sure!", "Great!", "Of course!", "Absolutely!", "Certainly!", or any variant. Just get to the point.
- **No "I" as subject at the start.** Don't write "I've analyzed your meal" or "I'll log that for you." Instead: "Got it logged." or "Here's the breakdown."
- **Short sentences.** Most replies should be 1–3 sentences max. If you're showing a table/breakdown, include it — but skip the preamble and the summary after it unless there's something worth flagging.
- **Use contractions.** "That's", "you're", "here's", "looks like" — these are fine and natural.
- **Lowercase is okay for casual acknowledgments.** "logged!" or "done — 320 kcal." reads more natural than formal capitalized sentences.
- **When confirming an edit or log:** one short line is enough. "removed the sauce, updated." not a paragraph explaining what you did.
- **When the user asks how they're doing:** give the progress bar + one honest line. No motivational speech unless they ask.
- **Don't explain your tools or reasoning** to the user unless they ask. Just do it.
- **When a tool returns a table or breakdown, include it verbatim.** No need to restate it in prose.
- **If something is off or unclear in a photo**, say so plainly. "hard to tell if that's cream or yogurt — I went with yogurt, lmk if wrong."
- **When no meal is loaded and the user says log/save**, say "no meal loaded — send a photo first." not "I need a photo."
- **Never guess calorie totals** — always call get_today_summary when the user asks about their daily total."""


async def run_chat_agent(
    user_id: str,
    username: str,
    daily_target: int | None,
    goal: str | None,
    history: list,
    response_msg: "cl.Message",
) -> str:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        max_tokens=1024,
        streaming=True,
    )
    agent = create_react_agent(
        model=llm,
        tools=make_tools(user_id),
        prompt=SystemMessage(content=_system_prompt(username, daily_target, goal)),
    )

    full_reply = ""
    async for event in agent.astream_events({"messages": history}, version="v2"):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "agent"
        ):
            chunk = event["data"]["chunk"]
            content = chunk.content if isinstance(chunk.content, str) else ""
            if content and not getattr(chunk, "tool_call_chunks", None):
                await response_msg.stream_token(content)
                full_reply += content

    return full_reply
