"""Recompute every user's face template with the model currently on disk.

Run this (with the server stopped) whenever you replace models/faceunlock_v1.keras by hand:
old templates only make sense for the model that made them. It re-enrolls each user from
their saved enrollment photos, all-or-nothing, and clears the cached dataset embeddings
used by the dashboard's face map.

    python scripts/reembed_users.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from app.db import get_db  # noqa: E402
from app.reenroll import reenroll_users  # noqa: E402


def main() -> None:
    app = create_app({"TESTING": True})  # TESTING: no heartbeat or warm-up threads
    with app.app_context():
        db = get_db()
        users = db.execute("SELECT id, username FROM users").fetchall()
        updated = reenroll_users(users, app.config["PHOTO_DIR"], app.config["VERIFIER"])
        db.executemany("UPDATE users SET embedding = ? WHERE id = ?", updated)
        db.commit()
    cleared = list(Path(app.config["INSIGHT_DIR"]).glob("dataset_v*.npz"))
    for cache in cleared:
        cache.unlink()
    print(f"re-enrolled {len(updated)} users: {', '.join(u['username'] for u in users)}")
    print(f"cleared {len(cleared)} cached dashboard embedding file(s)")


if __name__ == "__main__":
    main()
