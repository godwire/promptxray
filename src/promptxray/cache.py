"""Remember every model answer on disk.

Ablation asks the same model the same question many times across runs. Without
a cache, a second run costs the same as the first. The key is the exact prompt
text plus the model name, so any change to the prompt is a cache miss and
anything unchanged is free.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from .providers import Answer

DEFAULT_PATH = Path(".promptxray-cache.sqlite")


class Cache:
    def __init__(self, path: str | Path = DEFAULT_PATH, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = str(path)
        self.hits = 0
        self.misses = 0
        # check_same_thread=False + a lock: the runner asks the model from a
        # thread pool, and every one of those threads reads and writes here.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False) if enabled else None
        if self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS answers ("
                " key TEXT PRIMARY KEY, text TEXT, input_tokens INT, output_tokens INT)"
            )
            self._conn.commit()

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> Answer | None:
        if not self._conn:
            return None
        was_hit = False
        with self._lock:
            row = self._conn.execute(
                "SELECT text, input_tokens, output_tokens FROM answers WHERE key = ?",
                (self.key(model, prompt),),
            ).fetchone()
            was_hit = row is not None
            if was_hit:
                self.hits += 1
            else:
                self.misses += 1
        if row is None:
            return None
        return Answer(text=row[0], input_tokens=row[1], output_tokens=row[2])

    def put(self, model: str, prompt: str, answer: Answer) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
                (self.key(model, prompt), answer.text, answer.input_tokens, answer.output_tokens),
            )
            self._conn.commit()

    def clear(self) -> int:
        """Delete every cached answer; returns how many were removed."""
        if not self._conn:
            return 0
        with self._lock:
            cur = self._conn.execute("DELETE FROM answers")
            self._conn.commit()
        return cur.rowcount

    @property
    def stats(self) -> dict[str, int | None]:
        """Counts and a live size of the backing store."""
        if not self._conn:
            return {"entries": 0, "hits": self.hits, "misses": self.misses, "path": None}
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM answers").fetchone()
        return {"entries": count, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
