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


def test_control_auth_modes():
    """Header-first auth with constant-time compare; empty token = dev mode.

    Calls control_auth() directly — no TestClient, matching how this suite
    tests Orchestrator._in_scope and Scout. Importing main instantiates the
    module-level Orchestrator, so ./data/agent_state.db gets created in the
    working dir; harmless in CI since no threads start (start() runs only in
    the app lifespan or /resume).
    """
    from fastapi import HTTPException

    import main

    original = settings.control_token
    try:
        settings.control_token = ""  # dev mode: everything open
        assert main.control_auth(x_control_token=None, token=None) is True

        settings.control_token = "s3cret-token"
        assert main.control_auth(x_control_token="s3cret-token", token=None) is True
        assert main.control_auth(x_control_token=None, token="s3cret-token") is True

        with pytest.raises(HTTPException) as exc:
            main.control_auth(x_control_token="wrong", token=None)
        assert exc.value.status_code == 401

        with pytest.raises(HTTPException) as exc:
            main.control_auth(x_control_token=None, token=None)
        assert exc.value.status_code == 401
    finally:
        settings.control_token = original


def test_worker_model_is_valid_groq_id():
    """Guards against a repeat of the bare-'compound' 404: that ID shipped as
    the default and every worker call 404'd in production."""
    from config import settings

    valid = {
        "groq/compound",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
    assert settings.worker_model in valid


def test_scout_prompt_forces_delegation():
    """The template must tell the scout to delegate listicles to the worker,
    and must still format cleanly (the JSON-schema braces are escaped)."""
    from scout import USER_TEMPLATE

    rendered = USER_TEMPLATE.format(
        goal="find frameworks", url="https://example.com", title="t", text="c"
    )
    lowered = rendered.lower()
    assert "needs_worker" in lowered
    assert "listicle" in lowered
    assert "critical rule for delegation" in lowered
    # The schema example survived formatting with literal braces intact.
    assert '"needs_worker": true|false' in rendered

