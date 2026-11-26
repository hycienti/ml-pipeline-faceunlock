import sqlite3
import threading
import time
from contextlib import closing

KEEP_SECONDS = 7 * 24 * 3600


def start(app) -> threading.Thread:
    """Write a timestamp every HEARTBEAT_SECONDS so the dashboard can graph uptime.

    Run the app with a single worker process, or each worker writes its own beat.
    """

    def beat():
        while True:
            with closing(sqlite3.connect(app.config["DB_PATH"])) as conn:
                now = time.time()
                conn.execute("INSERT INTO heartbeats (ts) VALUES (?)", (now,))
                conn.execute("DELETE FROM heartbeats WHERE ts < ?", (now - KEEP_SECONDS,))
                conn.commit()
            time.sleep(app.config["HEARTBEAT_SECONDS"])

    thread = threading.Thread(target=beat, name="heartbeat", daemon=True)
    thread.start()
    return thread


def record_stop(app) -> None:
    """Mark a clean shutdown (idle scale-down, Ctrl-C). Time until the next start counts as up."""
    with closing(sqlite3.connect(app.config["DB_PATH"])) as conn:
        conn.execute("INSERT INTO app_stops (ts) VALUES (?)", (time.time(),))
        conn.commit()
