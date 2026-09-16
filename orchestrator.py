"""The autonomous loop: pick a URL, fetch it, ask the scout, optionally spend a
worker call, queue what's worth following, repeat.

On the open web the loop also runs *discovery* — search queries that inject
fresh entry points — because link-following alone can only ever reach the
neighbourhood of the original seed.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import structlog

from config import settings
from db import Database
from discovery import Discovery
from fetcher import Fetcher
from schemas import ScoutDecision
from scout import Scout
from worker import Worker

logger = structlog.get_logger()


class Orchestrator:
    def __init__(self) -> None:
        self.db = Database()
        self.fetcher = Fetcher()
        self.worker = Worker()
        self.scout: Optional[Scout] = None   # loaded lazily; ~1.5 GB RAM, slow init
        self.discovery = Discovery()
        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.pages_processed = 0
        self.discovery_runs = 0
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.running = True
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="orchestrator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def seed(self, urls: Optional[List[str]] = None) -> int:
        return sum(int(self.db.add_url(u)) for u in (urls or settings.seed_url_list))

    def run_discovery(self) -> int:
        """Search for new entry points and queue them at depth 0."""
        if not self.discovery.enabled:
            return 0
        self.discovery_runs += 1
        urls = self.discovery.discover()
        added = sum(int(self.db.add_url(u, depth=0)) for u in urls if self._in_scope(u))
        logger.info("discovery_queued", found=len(urls), added=added)
        return added

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "paused": self.paused,
            "scout_loaded": self.scout is not None,
            "worker_enabled": self.worker.enabled,
            "discovery_enabled": self.discovery.enabled,
            "discovery_provider": settings.search_provider,
            "open_web": settings.open_web,
            "goal": settings.current_goal,
            "pages_processed": self.pages_processed,
            "discovery_runs": self.discovery_runs,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "last_error": self.last_error,
            **self.db.stats(),
        }

    # --- main loop ---------------------------------------------------------
    def _run(self) -> None:
        logger.info("orchestrator_starting")
        try:
            self.scout = Scout()
            self.discovery.scout = self.scout
        except Exception as exc:
            self.last_error = f"scout load failed: {exc}"
            logger.error("scout_load_failed", error=str(exc))
            self.running = False
            return

        self.db.requeue_stale()
        self.seed()
        if self.db.stats()["queue_pending"] == 0:
            self.run_discovery()
        if self.db.stats()["queue_pending"] == 0:
            logger.warning(
                "no_starting_points",
                hint="set SEED_URLS, or SEARCH_PROVIDER + SEARCH_API_KEY for open-web discovery",
            )
        logger.info("orchestrator_started", goal=settings.current_goal, open_web=settings.open_web)

        while not self._stop_event.is_set():
            if self.paused:
                self._stop_event.wait(2.0)
                continue

            if settings.max_pages and self.pages_processed >= settings.max_pages:
                logger.info("max_pages_reached", pages=self.pages_processed)
                self.paused = True
                continue

            task = self.db.get_next_task(ready=self.fetcher.ready_for)
            if not task:
                self._idle()
                continue

            try:
                self._process_task(task)
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("task_failed", url=task.get("url"), error=str(exc))
            finally:
                self.db.complete_task(task["id"], task["url"])
                self.pages_processed += 1

            self._stop_event.wait(
                random.uniform(settings.min_delay_seconds, settings.max_delay_seconds)
            )

        self.running = False
        logger.info("orchestrator_stopped")

    def _idle(self) -> None:
        """Nothing claimable. Either every host is on cooldown, or we're dry."""
        if self.db.stats()["queue_pending"] == 0 and settings.search_on_empty_queue:
            if self.run_discovery():
                return
        nap = random.uniform(settings.idle_sleep_min, settings.idle_sleep_max)
        logger.info("idle_sleeping", seconds=round(nap, 1))
        self._stop_event.wait(nap)

    # --- single unit of work ----------------------------------------------
    def _process_task(self, task: Dict[str, Any]) -> None:
        url = task["url"]
        depth = task.get("depth", 0)

        if not self.fetcher.is_allowed(url):
            logger.info("skipped_by_robots", url=url)
            return

        page = self.fetcher.fetch_and_parse(url)
        decision: ScoutDecision = self.scout.decide(url, page.text, page.title)

        if not decision.relevant:
            decision.needs_worker = False
            decision.next_actions = [
                a for a in decision.next_actions if a.type != "call_worker"
            ]

        logger.info(
            "scout_decision",
            url=url,
            relevant=decision.relevant,
            needs_worker=decision.needs_worker,
            confidence=decision.confidence,
        )

        confident = decision.confidence >= settings.scout_min_confidence

        worker_result = None
        if decision.relevant and decision.needs_worker and confident:
            task_desc = next(
                (a.task for a in decision.next_actions if a.type == "call_worker" and a.task),
                settings.current_goal,
            )
            worker_result = self.worker.execute(url, page.text, task_desc)
            logger.info("worker_done", url=url, items=len(worker_result.items))

        if decision.relevant:
            self._queue_links(decision, page, depth)

        self.db.save_result(
            url=url,
            scout_dec=decision.model_dump(),
            worker_res=worker_result.model_dump() if worker_result else None,
            title=page.title,
        )

    def _queue_links(self, decision: ScoutDecision, page, depth: int) -> None:
        queued = 0
        candidates: List[str] = []
        # URLs the scout explicitly asked for go first.
        candidates += [a.url for a in decision.next_actions if a.type == "queue_url" and a.url]
        candidates += page.links

        seen_hosts: Dict[str, int] = {}
        for link in candidates:
            if queued >= settings.max_links_per_page:
                break
            if not self._in_scope(link) or self.db.domain_full(link):
                continue
            # Cap links per host per page so one nav menu can't fill the queue.
            host = (urlparse(link).hostname or "").lower()
            if seen_hosts.get(host, 0) >= 5:
                continue
            if self.db.add_url(link, depth=depth + 1):
                seen_hosts[host] = seen_hosts.get(host, 0) + 1
                queued += 1
        logger.info("links_queued", url=page.url, count=queued)

    @staticmethod
    def _in_scope(url: str) -> bool:
        if not Fetcher.is_fetchable(url):
            return False
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False

        def matches(domains) -> bool:
            return any(host == d or host.endswith("." + d) for d in domains)

        if matches(settings.blocked_domain_list):
            return False
        allowed = settings.allowed_domain_list
        return True if not allowed else matches(allowed)
