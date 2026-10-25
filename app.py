import base64
from datetime import datetime, timezone

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from backend.telemetry import setup_phoenix
from backend.agents.chat import run_chat_agent

setup_phoenix()
from backend.services import db
from backend.services.user import (
    get_user, create_partial_user, complete_profile,
    hash_password, verify_password,
)

_ACTIVITY_LABELS = {
    "sedentary":   "Sedentary (desk job, little exercise)",
    "light":       "Lightly active (1–3 days/week)",
    "moderate":    "Moderately active (3–5 days/week)",
    "active":      "Active (6–7 days/week)",
    "very_active": "Very active (physical job + exercise)",
}

_GOAL_LABELS = {
    "lose":     "Lose weight (−500 kcal/day)",
    "maintain": "Maintain weight",
    "gain":     "Gain weight (+300 kcal/day)",
}


# ── Auth ───────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
async def auth(username: str, password: str) -> cl.User | None:
    user = await get_user(username)

    if user:
        if not verify_password(password, user["password_hash"]):
            return None
        needs_onboarding = user.get("daily_target") is None
        return cl.User(
            identifier=username,
            metadata={
                "daily_target": user.get("daily_target"),
                "goal": user.get("goal"),
                "needs_onboarding": needs_onboarding,
            },
        )

    await create_partial_user(username, hash_password(password))
    return cl.User(identifier=username, metadata={"daily_target": None, "goal": None, "needs_onboarding": True})


# ── Chat start ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("meal", None)
    cl.user_session.set("pending_image", None)
    cl.user_session.set("history", [])

    username = cl.context.session.user.identifier

    # Always re-read from DB — JWT metadata can be stale after onboarding
    db_user = await get_user(username)
    if not db_user or db_user.get("daily_target") is None:
        await _run_onboarding(username)
        return

    await _show_daily_banner(username, db_user.get("daily_target"))


async def _run_onboarding(username: str):
    await cl.Message(
        content=f"Welcome, **{username}**! Let's set up your calorie target using a quick TDEE calculation."
    ).send()

    sex_res = await cl.AskActionMessage(
        content="What's your biological sex? (used for the BMR formula)",
        actions=[
            cl.Action(name="male", label="Male", payload={"value": "male"}),
            cl.Action(name="female", label="Female", payload={"value": "female"}),
        ],
    ).send()
    sex = (sex_res.get("payload") or {}).get("value", "male") if sex_res else "male"

    age = await _ask_int("How old are you? (years)", 10, 120)
    height_cm = await _ask_float("Your height in cm? (e.g. 175)", 100, 250)
    weight_kg = await _ask_float("Your weight in kg? (e.g. 70)", 30, 300)

    activity_res = await cl.AskActionMessage(
        content="What's your typical activity level?",
        actions=[cl.Action(name=k, label=v, payload={"value": k}) for k, v in _ACTIVITY_LABELS.items()],
    ).send()
    activity = (activity_res.get("payload") or {}).get("value", "moderate") if activity_res else "moderate"

    goal_res = await cl.AskActionMessage(
        content="What's your goal?",
        actions=[cl.Action(name=k, label=v, payload={"value": k}) for k, v in _GOAL_LABELS.items()],
    ).send()
    goal = (goal_res.get("payload") or {}).get("value", "maintain") if goal_res else "maintain"

    tdee, target = await complete_profile(username, sex, age, height_cm, weight_kg, activity, goal)
    adj = target - tdee
    adj_str = f"{adj:+,} kcal" if adj != 0 else "no adjustment"

    await cl.Message(
        content=(
            f"All set!\n\n"
            f"- **TDEE:** {tdee:,} kcal/day\n"
            f"- **Goal adjustment:** {adj_str}\n"
            f"- **Daily target: {target:,} kcal/day**\n\n"
            "Upload a photo to log your first meal."
        )
    ).send()


async def _show_daily_banner(username: str, daily_target: int | None):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = await db.query(
        "SELECT COALESCE(SUM(total_calories),0) AS total, COUNT(*) AS cnt FROM meals WHERE user_id=? AND created_at >= ?",
        [username, today],
    )
    today_kcal = round(float(rows[0]["total"])) if rows else 0
    meal_count = int(rows[0]["cnt"]) if rows else 0

    if daily_target:
        pct = min(today_kcal / daily_target, 1.0)
        bar = "█" * round(pct * 10) + "░" * (10 - round(pct * 10))
        remaining = max(daily_target - today_kcal, 0)
        banner = (
            f"Welcome back, **{username}**!\n\n"
            f"Today: **{today_kcal:,} / {daily_target:,} kcal**  {bar} {int(pct*100)}%\n"
            f"{meal_count} meal{'s' if meal_count != 1 else ''} logged · {remaining:,} kcal remaining\n\n"
            "Upload a photo to log your next meal."
        )
    else:
        banner = f"Welcome back, **{username}**! Upload a photo to log a meal."

    await cl.Message(content=banner).send()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        res = await cl.AskUserMessage(content=prompt).send()
        val = res.get("output", "").strip() if res else ""
        try:
            n = int(val)
            if lo <= n <= hi:
                return n
        except ValueError:
            pass
        await cl.Message(content=f"Please enter a whole number between {lo} and {hi}.").send()


async def _ask_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        res = await cl.AskUserMessage(content=prompt).send()
        val = res.get("output", "").strip() if res else ""
        try:
            n = float(val)
            if lo <= n <= hi:
                return n
        except ValueError:
            pass
        await cl.Message(content=f"Please enter a number between {lo} and {hi}.").send()


# ── Message handler ───────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    user_id = cl.context.session.user.identifier
    history: list = cl.user_session.get("history", [])

    db_user = await get_user(user_id)
    daily_target = int(db_user["daily_target"]) if db_user and db_user.get("daily_target") else None
    goal = db_user.get("goal") if db_user else None

    # Store image in session for the analyze_food tool to pick up
    if message.elements:
        image_el = next(
            (e for e in message.elements if getattr(e, "mime", "").startswith("image/")),
            None,
        )
        if image_el:
            with open(image_el.path, "rb") as f:
                cl.user_session.set("pending_image", {
                    "b64": base64.b64encode(f.read()).decode(),
                    "mime": image_el.mime or "image/jpeg",
                })

    user_content = message.content.strip() if message.content and message.content.strip() else "(uploaded a food photo)"
    history.append(HumanMessage(content=user_content))

    response_msg = cl.Message(content="")
    await response_msg.send()

    reply = await run_chat_agent(
        user_id=user_id,
        username=user_id,
        daily_target=daily_target,
        goal=goal,
        history=history,
        response_msg=response_msg,
    )

    if not reply:
        reply = "Done."
        await response_msg.stream_token(reply)

    await response_msg.update()
    history.append(AIMessage(content=reply))
    cl.user_session.set("history", history[-20:])
