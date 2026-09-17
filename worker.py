"""The expensive, capable extractor. Called only when the scout says a page is
worth it. Backed by Groq — default groq/compound-mini, JSON mode. Built-in
compound tools (web search etc.) are disabled: the page text is already here."""

from __future__ import annotations

import json

import structlog
from groq import Groq

from config import settings
from schemas import WorkerResult

logger = structlog.get_logger()

PROMPT = """You are a precise data extraction worker.
Task: {task}
Source URL: {url}

Extract the requested information from the content below. Do not invent
entries; if nothing matches, return an empty items list.

Reply with JSON exactly matching this schema:
{{"items": [{{"name": "string", "url": "string", "description": "string"}}],
  "summary": "string",
  "confidence": 0.0-1.0,
  "next_tasks": ["string"]}}

CONTENT:
{text}
"""


class Worker:
    def __init__(self) -> None:
        self.client = Groq(api_key=settings.worker_api_key) if settings.worker_api_key else None
        if not self.client:
            logger.warning("worker_disabled_no_api_key")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def execute(self, url: str, text: str, task_description: str) -> WorkerResult:
        if not self.client:
            return WorkerResult(summary="Skipped: WORKER_API_KEY not set")

        logger.info("worker_executing", url=url, task=task_description)
        # compound_custom disables the compound systems' built-in tools (web
        # search, code execution): this worker extracts from the text we give
        # it. Sent via extra_body so it works on any pinned groq SDK version;
        # plain-LLM models ignore the unknown field.
        try:
            completion = self.client.chat.completions.create(
                model=settings.worker_model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            task=task_description, url=url, text=text
                        ),
                    }
                ],
                temperature=0.1,
                max_tokens=settings.worker_max_tokens,
                response_format={"type": "json_object"},
                extra_body={"compound_custom": {"tools": {"enabled_tools": []}}},
            )
            data = json.loads(completion.choices[0].message.content)
            return WorkerResult(**data)
        except Exception as exc:
            logger.error("worker_failed", url=url, error=str(exc))
            return WorkerResult(summary=f"Worker error: {exc}")
