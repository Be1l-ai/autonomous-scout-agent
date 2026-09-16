"""Finding new places to look.

Pure link-following can only reach whatever the seed page happens to link to.
To actually explore the web the agent needs a discovery channel — a search API.
This module wraps a few, all optional: with SEARCH_PROVIDER=none the agent
still works, it just can't escape its seed neighbourhood.
"""

from __future__ import annotations

import json
from typing import List

import structlog
from curl_cffi import requests

from config import settings

logger = structlog.get_logger()


class Discovery:
    """Turns a goal into a list of candidate URLs."""

    def __init__(self, scout=None) -> None:
        self.scout = scout          # optional: used to invent queries from the goal
        self._used_queries: set[str] = set()

    @property
    def enabled(self) -> bool:
        provider = settings.search_provider.lower()
        if provider == "none":
            return False
        if provider == "searxng":
            return bool(settings.searxng_url)
        return bool(settings.search_api_key)

    # --- queries ------------------------------------------------------------
    def queries(self) -> List[str]:
        """Configured queries if given, otherwise ask the scout to invent some."""
        if settings.search_query_list:
            return settings.search_query_list
        if self.scout is None:
            return [settings.current_goal]
        try:
            generated = self.scout.generate_queries(settings.current_goal)
            return generated or [settings.current_goal]
        except Exception as exc:
            logger.warning("query_generation_failed", error=str(exc))
            return [settings.current_goal]

    # --- search -------------------------------------------------------------
    def search(self, query: str) -> List[str]:
        provider = settings.search_provider.lower()
        try:
            if provider == "brave":
                return self._brave(query)
            if provider == "tavily":
                return self._tavily(query)
            if provider == "searxng":
                return self._searxng(query)
        except Exception as exc:
            logger.error("search_failed", provider=provider, query=query, error=str(exc))
        return []

    def discover(self) -> List[str]:
        """Run every query and return a deduped list of URLs."""
        if not self.enabled:
            return []
        found: List[str] = []
        seen = set()
        for query in self.queries():
            if query in self._used_queries:
                continue
            self._used_queries.add(query)
            logger.info("searching", query=query, provider=settings.search_provider)
            for url in self.search(query):
                if url not in seen:
                    seen.add(url)
                    found.append(url)
        logger.info("discovery_complete", urls=len(found))
        return found

    # --- providers ----------------------------------------------------------
    def _brave(self, query: str) -> List[str]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": settings.search_results_per_query},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": settings.search_api_key,
            },
            timeout=settings.fetch_timeout,
        )
        resp.raise_for_status()
        return [r["url"] for r in resp.json().get("web", {}).get("results", []) if r.get("url")]

    def _tavily(self, query: str) -> List[str]:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.search_api_key,
                "query": query,
                "max_results": settings.search_results_per_query,
                "search_depth": "basic",
            },
            timeout=settings.fetch_timeout,
        )
        resp.raise_for_status()
        return [r["url"] for r in resp.json().get("results", []) if r.get("url")]

    def _searxng(self, query: str) -> List[str]:
        resp = requests.get(
            settings.searxng_url.rstrip("/") + "/search",
            params={"q": query, "format": "json"},
            timeout=settings.fetch_timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[: settings.search_results_per_query]
        return [r["url"] for r in results if r.get("url")]
