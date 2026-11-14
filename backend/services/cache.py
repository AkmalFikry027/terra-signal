"""SQLite cache of the last good dashboard payload per location (offline / fail-safe demo)."""
import json

from services.security import db


def save(location: str, payload: dict) -> None:
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO dashboard_cache(location,payload,updated_at) VALUES(?,?,?)",
                     (location, json.dumps(payload), payload["generated_at"]))


def load(location: str) -> dict | None:
    with db() as conn:
        row = conn.execute("SELECT payload, updated_at FROM dashboard_cache WHERE location=?",
                           (location,)).fetchone()
    return {"updated_at": row["updated_at"], "payload": json.loads(row["payload"])} if row else None
