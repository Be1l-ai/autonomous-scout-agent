"""FastAPI surface: health, control, and a tiny dashboard.

The orchestrator runs in a daemon thread. The model is loaded inside that
thread so the HTTP server answers /health immediately — important because HF
Spaces marks a Space unhealthy if the port doesn't bind fast enough.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse

from config import settings
from orchestrator import Orchestrator
from schemas import GoalUpdate, SeedRequest

logging.basicConfig(format="%(message)s", level=settings.log_level.upper())
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
)
logger = structlog.get_logger()

orchestrator = Orchestrator()


def control_auth(token: str = Query(None)):
    """Control plane authentication.
    
    If CONTROL_TOKEN is not set, allow access (dev mode).
    If token doesn't match, raise 401.
    """
    if not settings.control_token:
        return True  # dev mode
    if token != settings.control_token:
        raise HTTPException(status_code=401, detail="Invalid control token")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.autostart:
        orchestrator.start()
    yield
    orchestrator.stop()


app = FastAPI(title="Autonomous Scout Agent", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="control-token" content="{settings.control_token}">
<title>Scout Agent</title>
<style>
 body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1117;color:#c9d1d9;
      margin:0;padding:2rem;line-height:1.6}}
 h1{{color:#58a6ff;font-size:1.25rem;margin:0 0 1rem}}
 pre{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;overflow:auto}}
 a{{color:#58a6ff}} .row{{display:flex;gap:1rem;flex-wrap:wrap}}
 button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;
        padding:.4rem .8rem;cursor:pointer}}
 button:hover{{border-color:#58a6ff}}
</style></head><body>
<h1>Autonomous Scout Agent</h1>
<div class="row">
  <button onclick="ctl('pause')">pause</button>
  <button onclick="ctl('resume')">resume</button>
  <button onclick="ctl('seed')">re-seed</button>
  <button onclick="ctl('discover')">discover</button>
  <a href="/docs">api docs</a>
</div>
<h2 style="font-size:1rem">status</h2><pre id="s">loading…</pre>
<h2 style="font-size:1rem">findings</h2><pre id="r">loading…</pre>
<h2 style="font-size:1rem">domains</h2><pre id="d">loading…</pre>
<script>
function getToken() {{
  const meta = document.querySelector('meta[name="control-token"]');
  return meta ? meta.getAttribute('content') : '';
}}
async function tick(){{
  try{{
    document.getElementById('s').textContent =
      JSON.stringify(await (await fetch('/stats')).json(),null,2);
    document.getElementById('r').textContent =
      JSON.stringify(await (await fetch('/results?only_with_items=true&limit=10')).json(),null,2);
    document.getElementById('d').textContent =
      JSON.stringify(await (await fetch('/domains?limit=15')).json(),null,2);
  }}catch(e){{}}
}}
async function ctl(p){{
  const token = getToken();
  const url = token ? '/' + p + '?token=' + encodeURIComponent(token) : '/' + p;
  await fetch(url, {{method:'POST'}});
  tick();
}}
tick(); setInterval(tick,5000);
</script></body></html>"""


@app.get("/health")
def health():
    """Kept deliberately cheap — this is the URL your uptime monitor pings."""
    return {
        "status": "alive",
        "goal": settings.current_goal,
        "running": orchestrator.running,
        "scout_loaded": orchestrator.scout is not None,
    }


@app.get("/stats")
def stats():
    return orchestrator.status()


@app.get("/results")
def results(limit: int = 20, only_with_items: bool = False):
    limit = max(1, min(limit, 200))
    return orchestrator.db.recent_results(limit=limit, only_with_items=only_with_items)


@app.get("/queue")
def queue(limit: int = 50):
    return orchestrator.db.pending(limit=max(1, min(limit, 200)))


@app.get("/domains")
def domains(limit: int = 20):
    """Where the crawl budget is actually going."""
    return orchestrator.db.top_domains(limit=max(1, min(limit, 200)))


@app.post("/discover")
def discover(_: bool = Depends(control_auth)):
    """Run the search queries now and queue whatever comes back."""
    if not orchestrator.discovery.enabled:
        raise HTTPException(
            status_code=400,
            detail="Discovery disabled. Set SEARCH_PROVIDER (brave|tavily|searxng) "
                   "and SEARCH_API_KEY.",
        )
    return {"added": orchestrator.run_discovery()}


@app.post("/pause")
def pause(_: bool = Depends(control_auth)):
    orchestrator.pause()
    return {"paused": True}


@app.post("/resume")
def resume(_: bool = Depends(control_auth)):
    if not orchestrator.running:
        orchestrator.start()
    orchestrator.resume()
    return {"paused": False, "running": orchestrator.running}


@app.post("/seed")
def seed(req: SeedRequest | None = None, _: bool = Depends(control_auth)):
    urls = req.urls if req and req.urls else None
    return {"added": orchestrator.seed(urls)}


@app.post("/goal")
def set_goal(update: GoalUpdate, _: bool = Depends(control_auth)):
    if not update.goal.strip():
        raise HTTPException(status_code=400, detail="goal must not be empty")
    settings.current_goal = update.goal.strip()
    logger.info("goal_updated", goal=settings.current_goal)
    return {"goal": settings.current_goal}


@app.post("/worker/test")
def worker_test(url: str, _: bool = Depends(control_auth)):
    """Smoke test: fetch a URL and run the worker on it."""
    fd = orchestrator.fetcher.fetch_and_parse(url)
    res = orchestrator.worker.execute(url, fd.text[:8000], settings.current_goal)
    orchestrator.db.save_result(url, {"manual_test": True}, res.model_dump())
    return res.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860)
