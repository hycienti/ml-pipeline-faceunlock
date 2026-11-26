import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    embedding BLOB NOT NULL,
    model_version INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_events (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL COLLATE NOCASE,
    success INTEGER NOT NULL,
    score REAL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS heartbeats (ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS heartbeats_ts ON heartbeats (ts);
CREATE TABLE IF NOT EXISTS app_starts (ts REAL NOT NULL);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    get_db().executescript(SCHEMA)
