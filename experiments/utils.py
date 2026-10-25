"""
Shared Phoenix startup helper for standalone experiment scripts.
Sets PHOENIX_WORKING_DIR to an absolute path so data persists across runs,
then waits until the HTTP server is ready before returning a Client.
"""
import os
import time
from pathlib import Path

PHOENIX_DIR = Path(__file__).parent.parent / ".phoenix"


def ensure_phoenix():
    """Launch Phoenix if not already running. Blocks until the server is ready."""
    import phoenix as px

    os.environ["PHOENIX_WORKING_DIR"] = str(PHOENIX_DIR)
    PHOENIX_DIR.mkdir(exist_ok=True)

    if px.active_session() is not None:
        return

    px.launch_app()

    import httpx
    for _ in range(30):
        try:
            httpx.get("http://localhost:6006/v1/datasets", timeout=1.0)
            return
        except Exception:
            time.sleep(0.3)
