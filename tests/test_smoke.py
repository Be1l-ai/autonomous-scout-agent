import os
import tempfile

import pytest

from config import settings
from db import Database
from schemas import ScoutDecision, WorkerResult


@pytest.fixture()
def db():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    database = Database(path=path)
    yield database
    database.close()


def test_queue_roundtrip(db):
    assert db.add_url("https://example.com/a") is True
    task = db.get_next_task()
    assert task["url"] == "https://example.com/a"
    assert db.get_next_task() is None  # now 'running', not 'pending'
    db.complete_task(task["id"], task["url"])
    assert db.stats()["queue_pending"] == 0


def test_urls_are_not_recrawled(db):
    db.add_url("https://example.com/a")
    task = db.get_next_task()
    db.complete_task(task["id"], task["url"])
    assert db.add_url("https://example.com/a") is False


def test_depth_cap(db):
    assert db.add_url("https://example.com/deep", depth=settings.max_depth + 1) is False


def test_per_domain_cap(db):
    original = settings.max_pages_per_domain
    settings.max_pages_per_domain = 2
    try:
        for i in range(2):
            db.add_url(f"https://busy.com/{i}")
            task = db.get_next_task()
            db.complete_task(task["id"], task["url"])
        assert db.add_url("https://busy.com/3") is False
        assert db.add_url("https://other.com/1") is True
    finally:
        settings.max_pages_per_domain = original


def test_least_crawled_domain_goes_first(db):
    """Breadth: a host we've already hammered should sort behind a fresh one."""
    db.add_url("https://hot.com/seen")
    task = db.get_next_task()
    db.complete_task(task["id"], task["url"])

    db.add_url("https://hot.com/next")
    db.add_url("https://fresh.com/next")
    assert db.next_candidates()[0]["url"] == "https://fresh.com/next"


def test_requeue_stale(db):
    db.add_url("https://example.com/a")
    db.get_next_task()
    assert db.requeue_stale() == 1
    assert db.stats()["queue_pending"] == 1


def test_save_and_read_results(db):
    db.save_result(
        url="https://example.com/a",
        scout_dec=ScoutDecision(relevant=True, confidence=0.9).model_dump(),
        worker_res=WorkerResult(summary="found things").model_dump(),
        title="Example",
    )
    rows = db.recent_results(limit=5)
    assert rows[0]["url"] == "https://example.com/a"
    assert rows[0]["worker_result"]["summary"] == "found things"
    assert rows[0]["domain"] == "example.com"


def test_scope_open_web_blocks_only_blocklist():
    from orchestrator import Orchestrator

    original = settings.allowed_domains
    settings.allowed_domains = ""
    try:
        assert Orchestrator._in_scope("https://some-random-blog.dev/post") is True
        assert Orchestrator._in_scope("https://www.facebook.com/x") is False
        assert Orchestrator._in_scope("https://cdn.site.com/video.mp4") is False
        assert Orchestrator._in_scope("ftp://site.com/x") is False
    finally:
        settings.allowed_domains = original


def test_scope_allowlist():
    from orchestrator import Orchestrator

    original = settings.allowed_domains
    settings.allowed_domains = "github.com"
    try:
        assert Orchestrator._in_scope("https://github.com/x") is True
        assert Orchestrator._in_scope("https://gist.github.com/x") is True
        assert Orchestrator._in_scope("https://evil-github.com/x") is False
        assert Orchestrator._in_scope("https://example.com/x") is False
    finally:
        settings.allowed_domains = original


def test_domain_throttle():
    from fetcher import DomainThrottle

    throttle = DomainThrottle()
    assert throttle.ready("example.com") is True
    throttle.mark("example.com")
    assert throttle.ready("example.com") is False
    assert throttle.ready("other.com") is True


def test_discovery_disabled_without_key():
    from discovery import Discovery

    original = settings.search_provider
    settings.search_provider = "brave"
    settings.search_api_key = ""
    try:
        assert Discovery().enabled is False
        settings.search_api_key = "fake"
        assert Discovery().enabled is True
    finally:
        settings.search_provider = original
        settings.search_api_key = ""


def test_scout_falls_back_on_bad_json():
    import threading

    from scout import Scout

    scout = Scout.__new__(Scout)  # skip __init__ / model load

    class Bad:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "not json at all"}}]}

    scout.llm = Bad()
    scout._lock = threading.Lock()
    decision = scout.decide("https://example.com", "text")
    assert decision.relevant is False
    assert decision.confidence == 0.0
