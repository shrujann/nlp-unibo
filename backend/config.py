import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash-lite")
CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", "deepseek/deepseek-v4-flash")
