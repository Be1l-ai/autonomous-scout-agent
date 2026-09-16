from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ScoutAction(BaseModel):
    type: Literal["queue_url", "call_worker", "stop"]
    url: Optional[str] = None
    task: Optional[str] = None


class ScoutDecision(BaseModel):
    relevant: bool = False
    needs_worker: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    next_actions: List[ScoutAction] = Field(default_factory=list)


class WorkerItem(BaseModel):
    name: str = ""
    url: str = ""
    description: str = ""


class WorkerResult(BaseModel):
    items: List[WorkerItem] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    next_tasks: List[str] = Field(default_factory=list)


class FetchResult(BaseModel):
    url: str
    status_code: int
    text: str
    links: List[str] = Field(default_factory=list)
    title: str = ""


class GoalUpdate(BaseModel):
    goal: str


class SeedRequest(BaseModel):
    urls: List[str]
