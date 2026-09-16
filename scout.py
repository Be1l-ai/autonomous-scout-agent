"""The local, cheap decision-maker. Runs Qwen2.5-1.5B-Instruct on CPU via
llama-cpp-python and answers one question per page: is this worth spending a
paid worker call on?"""

from __future__ import annotations

import json
import threading
from typing import List

import structlog
from llama_cpp import Llama

from config import settings
from schemas import ScoutDecision

logger = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are an autonomous scouting agent. You triage web pages against a goal. "
    "You always reply with a single valid JSON object and nothing else."
)

USER_TEMPLATE = """Current goal: {goal}

Decide whether this page is relevant to the goal, and whether it needs the
expensive extraction worker (only when the page actually contains the data we
want, not just links to it).

Reply with JSON exactly matching this schema:
{{"relevant": true|false,
  "needs_worker": true|false,
  "confidence": 0.0-1.0,
  "reason": "one short sentence",
  "next_actions": [{{"type": "queue_url"|"call_worker"|"stop", "url": null, "task": null}}]}}

URL: {url}
TITLE: {title}
CONTENT:
{text}
"""

QUERY_TEMPLATE = """Goal: {goal}

Write {n} short, varied web search queries that would surface pages relevant to
this goal. Use different angles and vocabulary; do not repeat the goal verbatim.

Reply with JSON exactly matching: {{"queries": ["...", "..."]}}
"""


class Scout:
    def __init__(self) -> None:
        model_path = settings.resolve_model_path()
        logger.info("loading_scout_model", path=model_path)
        # n_gpu_layers=0 forces CPU. n_threads is deliberately small so the
        # FastAPI event loop and the host OS aren't starved on a 2-vCPU box.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=settings.scout_context_size,
            n_threads=settings.scout_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        self._lock = threading.Lock()
        logger.info("scout_model_loaded")

    def decide(self, url: str, text: str, title: str = "") -> ScoutDecision:
        prompt = USER_TEMPLATE.format(
            goal=settings.current_goal, url=url, title=title, text=text
        )
        logger.info("scout_thinking", url=url)

        try:
            with self._lock:  # llama.cpp contexts are not thread-safe
                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=settings.scout_temperature,
                    max_tokens=settings.scout_max_tokens,
                    response_format={"type": "json_object"},
                )
            raw = response["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("scout_inference_failed", url=url, error=str(exc))
            return ScoutDecision(reason=f"Scout inference error: {exc}")

        logger.debug("scout_raw_output", raw=raw)
        try:
            return ScoutDecision(**json.loads(raw))
        except Exception as exc:
            logger.error("scout_json_parse_error", error=str(exc), raw=raw[:400])
            return ScoutDecision(reason="Failed to parse scout JSON")

    def generate_queries(self, goal: str, n: int = 4) -> List[str]:
        """Turn the goal into web search queries, so discovery isn't limited to
        the literal goal string."""
        prompt = QUERY_TEMPLATE.format(goal=goal, n=n)
        try:
            with self._lock:
                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
            data = json.loads(response["choices"][0]["message"]["content"])
            queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
            logger.info("queries_generated", queries=queries)
            return queries[:n]
        except Exception as exc:
            logger.warning("query_generation_failed", error=str(exc))
            return []
