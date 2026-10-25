import os
import httpx
from typing import Any


def _d1_url() -> str:
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    db_id = os.environ["CF_D1_DATABASE_ID"]
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{db_id}/query"


async def query(sql: str, params: list[Any] | None = None) -> list[dict]:
    url = _d1_url()
    token = os.environ["CLOUDFLARE_API_TOKEN"]

    payload: dict[str, Any] = {"sql": sql}
    if params:
        payload["params"] = params

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

    if not data.get("success"):
        errors = data.get("errors", [])
        raise RuntimeError(f"D1 query failed: {errors}")

    results = data.get("result", [])
    if not results:
        return []

    return results[0].get("results", [])
