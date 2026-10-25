import hashlib
import secrets
from typing import Optional

from backend.services import db

_ACTIVITY_MULTIPLIERS = {
    "sedentary":   1.20,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.90,
}

_GOAL_ADJUSTMENTS = {
    "lose":     -500,
    "maintain": 0,
    "gain":     300,
}


def calculate_tdee(sex: str, age: int, height_cm: float, weight_kg: float, activity: str) -> int:
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    bmr += 5 if sex == "male" else -161
    return round(bmr * _ACTIVITY_MULTIPLIERS[activity])


def calculate_target(tdee: int, goal: str) -> int:
    return tdee + _GOAL_ADJUSTMENTS[goal]


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"sha256${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == digest
    except Exception:
        return False


async def get_user(username: str) -> Optional[dict]:
    rows = await db.query("SELECT * FROM users WHERE id = ?", [username])
    return rows[0] if rows else None


async def create_partial_user(username: str, password_hash: str) -> None:
    await db.query(
        "INSERT OR IGNORE INTO users (id, password_hash) VALUES (?, ?)",
        [username, password_hash],
    )


async def complete_profile(
    username: str,
    sex: str,
    age: int,
    height_cm: float,
    weight_kg: float,
    activity: str,
    goal: str,
) -> tuple[int, int]:
    tdee = calculate_tdee(sex, age, height_cm, weight_kg, activity)
    target = calculate_target(tdee, goal)
    await db.query(
        """UPDATE users SET sex=?, age=?, height_cm=?, weight_kg=?, activity=?, goal=?, tdee=?, daily_target=?
           WHERE id=?""",
        [sex, age, height_cm, weight_kg, activity, goal, tdee, target, username],
    )
    return tdee, target
