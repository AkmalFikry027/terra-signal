"""SQLite-backed users, PBKDF2 password hashing, signed tokens and login throttling."""
import base64
import hashlib
import hmac
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import jwt

import config

_ITERATIONS = 310_000


@contextmanager
def db():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str, salt: bytes | None = None, iterations: int = _ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"pbkdf2_sha256${iterations}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iters))
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except Exception:
        return False


_DUMMY_HASH = hash_password(secrets.token_hex(8))  # equalises timing for unknown usernames


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            email TEXT NOT NULL,
            token_version INTEGER NOT NULL DEFAULT 1,
            is_default_password INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS dashboard_cache (
            location TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """)
        if not conn.execute("SELECT 1 FROM users WHERE username=?", (config.DEFAULT_USERNAME,)).fetchone():
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                (config.DEFAULT_USERNAME, hash_password(config.DEFAULT_PASSWORD),
                 config.AUTHOR["name"], config.AUTHOR["email"], 1, 1,
                 datetime.now(timezone.utc).isoformat()),
            )
        if not conn.execute("SELECT 1 FROM meta WHERE key='jwt_secret'").fetchone():
            conn.execute("INSERT INTO meta VALUES ('jwt_secret', ?)", (secrets.token_hex(32),))


def _secret() -> str:
    with db() as conn:
        return conn.execute("SELECT value FROM meta WHERE key='jwt_secret'").fetchone()["value"]


def get_user(username: str):
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def public_user(row) -> dict:
    return {"username": row["username"], "display_name": row["display_name"], "email": row["email"],
            "is_default_password": bool(row["is_default_password"])}


def create_token(row) -> tuple[str, int]:
    ttl = timedelta(hours=config.TOKEN_TTL_HOURS)
    payload = {"sub": row["username"], "tv": row["token_version"],
               "exp": datetime.now(timezone.utc) + ttl}
    return jwt.encode(payload, _secret(), algorithm="HS256"), int(ttl.total_seconds())


def user_from_token(token: str):
    try:
        data = jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    row = get_user(data.get("sub", ""))
    if row is None or row["token_version"] != data.get("tv"):
        return None  # password changed -> older tokens are revoked
    return row


# --- login throttling (per username + client IP, in memory) -----------------
_failures: dict[str, deque] = defaultdict(deque)


def _key(username: str, ip: str) -> str:
    return f"{username.lower()}|{ip}"


def lock_remaining(username: str, ip: str) -> int:
    q = _failures[_key(username, ip)]
    now = time.time()
    while q and now - q[0] > config.LOGIN_LOCK_SECONDS:
        q.popleft()
    if len(q) >= config.LOGIN_MAX_FAILURES:
        return int(config.LOGIN_LOCK_SECONDS - (now - q[0])) + 1
    return 0


def authenticate(username: str, password: str, ip: str):
    row = get_user(username)
    ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
    if row and ok:
        _failures.pop(_key(username, ip), None)
        return row
    _failures[_key(username, ip)].append(time.time())
    return None


def change_password(row, new_password: str) -> dict:
    with db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, token_version=token_version+1, "
            "is_default_password=0, updated_at=? WHERE username=?",
            (hash_password(new_password), datetime.now(timezone.utc).isoformat(), row["username"]),
        )
    return get_user(row["username"])
