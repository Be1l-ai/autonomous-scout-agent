"""Ephemeral state + work queue.

On HF free-tier Spaces the container filesystem is wiped on restart, so treat
this as a cache, not a datastore. To make it durable, point `db_path` at a
mounted volume, or swap the sqlite3 calls for libSQL/Turso or Postgres — the
public method signatures below are the only surface the rest of the app uses.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import structlog

from config import settings

logger = structlog.get_logger()


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


class Database:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or settings.db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS queue (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    domain TEXT,
                    type TEXT,
                    status TEXT,
                    depth INTEGER DEFAULT 0,
                    created_at REAL,
                    next_attempt_at REAL DEFAULT 0,
                    priority INTEGER DEFAULT 5
                );
                -- `seen` is permanent: queue rows are deleted on completion, so
                -- without this the crawler would happily re-visit forever.
                CREATE TABLE IF NOT EXISTS seen (
                    url TEXT PRIMARY KEY,
                    created_at REAL
                );
                -- Drives both the per-domain cap and breadth-first host rotation.
                CREATE TABLE IF NOT EXISTS domains (
                    domain TEXT PRIMARY KEY,
                    pages INTEGER DEFAULT 0,
                    last_seen REAL
                );
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    url TEXT,
                    domain TEXT,
                    title TEXT,
                    scout_decision TEXT,
                    worker_result TEXT,
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
                CREATE INDEX IF NOT EXISTS idx_queue_domain ON queue(domain);
                CREATE INDEX IF NOT EXISTS idx_results_created ON results(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_queue_next_attempt ON queue(next_attempt_at);
                CREATE INDEX IF NOT EXISTS idx_queue_priority ON queue(priority);
                """
            )
            self.conn.commit()

# --- queue -------------------------------------------------------------
    def add_url(self, url: str, task_type: str = "FETCH_URL", depth: int = 0, priority: int = 5) -> bool:
        if settings.max_depth and depth > settings.max_depth:
            return False
        domain = host_of(url)
        with self._lock:
            cur = self.conn.cursor()
            if cur.execute("SELECT 1 FROM seen WHERE url=?", (url,)).fetchone():
                return False
            if settings.max_queue_size:
                pending = cur.execute(
                    "SELECT COUNT(*) FROM queue WHERE status='pending'"
                ).fetchone()[0]
                if pending >= settings.max_queue_size:
                    return False
            if settings.max_pages_per_domain:
                row = cur.execute(
                    "SELECT pages FROM domains WHERE domain=?", (domain,)
                ).fetchone()
                if row and row["pages"] >= settings.max_pages_per_domain:
                    return False
            cur.execute(
                "INSERT OR IGNORE INTO queue (id, url, domain, type, status, depth, created_at, next_attempt_at, priority) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, 0, ?)",
                (str(uuid.uuid4()), url, domain, task_type, depth, time.time(), priority),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def next_candidates(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Pending tasks, least-crawled hosts first.

        Ordering by domain page-count is what keeps an open-web crawl broad: a
        plain FIFO queue degenerates into depth-first on whichever site links
        the most, because that site's links dominate the queue.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT q.id, q.url, q.type, q.depth, q.priority, COALESCE(d.pages, 0) AS domain_pages
                FROM queue q
                LEFT JOIN domains d ON d.domain = q.domain
                WHERE q.status='pending'
                ORDER BY q.priority ASC, domain_pages ASC, q.depth ASC, q.created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def claim(self, task_id: str) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE queue SET status='running' WHERE id=? AND status='pending'",
                (task_id,),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def get_next_task(
        self, ready: Optional[Callable[[str], bool]] = None
    ) -> Optional[Dict[str, Any]]:
        """Claim the best pending task whose host is off cooldown."""
        now = time.time()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT q.id, q.url, q.type, q.depth, q.priority, COALESCE(d.pages, 0) AS domain_pages
                FROM queue q
                LEFT JOIN domains d ON d.domain = q.domain
                WHERE q.status='pending' AND q.next_attempt_at <= ?
                ORDER BY q.priority ASC, domain_pages ASC, q.depth ASC, q.created_at ASC
                LIMIT 25
                """,
                (now,),
            ).fetchall()
        for task in (dict(r) for r in rows):
            if ready and not ready(task["url"]):
                continue
            if self.claim(task["id"]):
                return task
        return None

    def requeue_later(self, url: str, seconds: int) -> bool:
        """Requeue a URL with a delayed next_attempt_at."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE queue SET status='pending', next_attempt_at=? WHERE url=?",
                (time.time() + seconds, url),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def complete_task(self, task_id: str, url: Optional[str] = None) -> None:
        with self._lock:
            cur = self.conn.cursor()
            if url:
                cur.execute(
                    "INSERT OR IGNORE INTO seen (url, created_at) VALUES (?, ?)",
                    (url, time.time()),
                )
                cur.execute(
                    "INSERT INTO domains (domain, pages, last_seen) VALUES (?, 1, ?) "
                    "ON CONFLICT(domain) DO UPDATE SET pages=pages+1, last_seen=excluded.last_seen",
                    (host_of(url), time.time()),
                )
            cur.execute("DELETE FROM queue WHERE id=?", (task_id,))
            self.conn.commit()

    def domain_full(self, url: str) -> bool:
        if not settings.max_pages_per_domain:
            return False
        with self._lock:
            row = self.conn.execute(
                "SELECT pages FROM domains WHERE domain=?", (host_of(url),)
            ).fetchone()
        return bool(row and row["pages"] >= settings.max_pages_per_domain)

    def requeue_stale(self) -> int:
        """Reset rows left in 'running' by a crashed/restarted process."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("UPDATE queue SET status='pending' WHERE status='running'")
            self.conn.commit()
            return cur.rowcount

    def pending(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT url, domain, type, status, depth, created_at FROM queue "
                "ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def top_domains(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT domain, pages FROM domains ORDER BY pages DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- results -----------------------------------------------------------
    def save_result(
        self,
        url: str,
        scout_dec: Dict[str, Any],
        worker_res: Optional[Dict[str, Any]] = None,
        title: str = "",
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO results (id, url, domain, title, scout_decision, worker_result, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    url,
                    host_of(url),
                    title,
                    json.dumps(scout_dec),
                    json.dumps(worker_res) if worker_res else None,
                    time.time(),
                ),
            )
            self.conn.commit()

    def recent_results(
        self, limit: int = 20, only_with_items: bool = False
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT url, domain, title, scout_decision, worker_result, created_at "
                "FROM results ORDER BY created_at DESC LIMIT ?",
                (limit * 6 if only_with_items else limit,),
            ).fetchall()

        out = []
        for row in (dict(r) for r in rows):
            row["scout_decision"] = json.loads(row["scout_decision"] or "{}")
            row["worker_result"] = json.loads(row["worker_result"] or "null")
            if only_with_items and not (row["worker_result"] or {}).get("items"):
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def stats(self) -> Dict[str, int]:
        def count(sql: str) -> int:
            return self.conn.execute(sql).fetchone()[0]

        with self._lock:
            return {
                "queue_pending": count("SELECT COUNT(*) FROM queue WHERE status='pending'"),
                "queue_running": count("SELECT COUNT(*) FROM queue WHERE status='running'"),
                "urls_seen": count("SELECT COUNT(*) FROM seen"),
                "domains_visited": count("SELECT COUNT(*) FROM domains"),
                "results": count("SELECT COUNT(*) FROM results"),
                "worker_runs": count("SELECT COUNT(*) FROM results WHERE worker_result IS NOT NULL"),
            }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
