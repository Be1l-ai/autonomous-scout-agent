---
title: Autonomous Scout Agent
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Autonomous Scout / Worker Agent

A two-tier autonomous web agent. A small local model triages every page for
free; a larger hosted model is only paid for when the local one says a page is
actually worth extracting.

```
seed URL ──> fetcher ──> SCOUT (Qwen2.5-1.5B, local CPU)
                            │
              relevant? ────┼──── no  ──> drop
                            │
                            ├── yes, needs extraction ──> WORKER (Groq compound-mini) ──> results
                            └── yes, just navigation  ──> queue discovered links
```

| Layer | Model | Runs on | Cost |
|---|---|---|---|
| Scout | Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) | local CPU, llama.cpp | free |
| Worker | groq/compound-mini (JSON mode, built-in tools off) | Groq API | free tier |

## API keys

| Key | Needed? | Where | Free tier |
|---|---|---|---|
| `WORKER_API_KEY` | **yes** | [console.groq.com/keys](https://console.groq.com/keys) | generous |
| `GITHUB_TOKEN` | no — but speeds up item link resolution | [github.com/settings/tokens](https://github.com/settings/tokens) (no scopes) | 30 search req/min vs 10 |
| `SEARCH_API_KEY` | only for open-web exploration | Brave or Tavily (below) | 1–2k queries/mo |

Without `WORKER_API_KEY` the agent still crawls and triages, but every page
comes back `"Skipped: WORKER_API_KEY not set"` — nothing gets extracted.

The scout needs no key at all; it runs locally.

## Quick start (local)

```bash
git clone <your-repo-url> agent-scout && cd agent-scout
python -m venv .venv && source .venv/bin/activate

cp .env.example .env          # then paste your Groq key into WORKER_API_KEY
pip install -r requirements.txt \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

python download_model.py      # ~1.1 GB
uvicorn main:app --port 7860 --reload
```

Open http://127.0.0.1:7860 for the dashboard, or http://127.0.0.1:7860/docs for
the API.

With Docker:

```bash
docker build -t agent-scout .
docker run -p 7860:7860 -e WORKER_API_KEY=gsk_... agent-scout
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Live dashboard |
| GET | `/health` | Cheap liveness check — point your uptime monitor here |
| GET | `/stats` | Queue depth, pages processed, worker calls, last error |
| GET | `/results?limit=20&only_with_items=true` | What the agent has found |
| GET | `/queue` | Pending URLs |
| POST | `/pause` · `/resume` | Control the loop |
| POST | `/seed` | `{"urls": ["https://..."]}` — inject new starting points |
| POST | `/goal` | `{"goal": "..."}` — retarget the agent at runtime |
| GET | `/domains` | Pages spent per host — where the budget is going |
| POST | `/discover` | Run search queries now, queue the results |

## Exploring the open web

`ALLOWED_DOMAINS` defaults to empty, which means no domain restriction — the
agent will follow links anywhere. Nothing in the code is site-specific.

But an unfenced crawler needs a discovery channel, or it just wanders the
neighbourhood its seed happens to link into. Set one up:

| Provider | Key | Free tier | Notes |
|---|---|---|---|
| **Brave Search** | [brave.com/search/api](https://brave.com/search/api/) | 2,000 queries/mo | good default; card required, not charged |
| **Tavily** | [tavily.com](https://tavily.com) | 1,000 credits/mo | built for agents, no card |
| **SearXNG** | none | unlimited | self-hosted; `SEARXNG_URL=` your instance |

```bash
SEARCH_PROVIDER=brave
SEARCH_API_KEY=BSA...
ALLOWED_DOMAINS=
CURRENT_GOAL=Find open-source AI agent frameworks and their source repositories.
```

The scout writes the search queries itself from the goal (a few varied ones per
round, not the goal verbatim), runs discovery on startup and again whenever the
queue runs dry, and queues the results at depth 0. Change `CURRENT_GOAL` at
runtime via `POST /goal` and the next discovery round follows the new goal. Set
`SEARCH_QUERIES` if you'd rather pin the queries yourself.

Force a round any time with `POST /discover`.

### What keeps an open crawl from exploding

A naive FIFO queue plus link-following degenerates within an hour: one link-rich
site fills the queue with its own URLs and the crawl becomes depth-first on a
single domain. Four things prevent that, all tunable:

- **`MAX_DEPTH=3`** — hops from a seed. The web branches ~20x per page, so
  depth 5 is already millions of URLs.
- **`MAX_PAGES_PER_DOMAIN=25`** — hard ceiling per host, so no single site can
  absorb the budget.
- **Least-crawled-host-first ordering** — `Database.next_candidates` sorts
  pending URLs by how many pages that host has already cost, which keeps the
  frontier broad instead of deep.
- **`BLOCKED_DOMAINS`** — ad networks, social logins, marketplaces, video
  platforms. See `DEFAULT_BLOCKLIST` in `config.py`.

Plus `PER_DOMAIN_DELAY_SECONDS=15`, a per-host cooldown enforced independently
of the global jitter, and binary/media URLs skipped before they're ever fetched.

On 2 free vCPUs the scout takes roughly 5–15s per page, so expect a few hundred
pages a day. This is a slow, broad explorer, not a bulk scraper.

## Configuration

Every field in `config.py` is settable via an environment variable of the same
name in upper case. See `.env.example` for the ones you'll actually touch.

The settings that matter most on a free tier:

- `SCOUT_THREADS=2` — more threads on a 2-vCPU box starves the web server and
  can get a Space flagged for abusive CPU use.
- `SEARCH_PROVIDER` / `SEARCH_API_KEY` — the difference between exploring the
  web and circling one seed.
- `ALLOWED_DOMAINS` — empty for the open web; set host suffixes to fence the
  agent into specific sites.
- `MAX_DEPTH`, `MAX_PAGES_PER_DOMAIN` — the two knobs that decide breadth.
- `MIN_DELAY_SECONDS` / `MAX_DELAY_SECONDS` / `PER_DOMAIN_DELAY_SECONDS` —
  politeness.
- `RESPECT_ROBOTS=true` — leave this on.

## Deploying to Hugging Face Spaces

1. Create a new **Docker** Space (blank template).
2. Settings → *Variables and secrets*:
   - Secret `WORKER_API_KEY` = your Groq key
   - Variable `CURRENT_GOAL` = whatever you want it hunting for
   - Variable `ALLOWED_DOMAINS` = leave empty for the open web, or e.g. `github.com`
   - Secret `SEARCH_API_KEY` + variable `SEARCH_PROVIDER` = `brave` (for discovery)
3. Push this repo to the Space remote (see below). The build downloads the
   GGUF weights into the image, so first boot is fast; the build itself takes
   roughly 10 minutes.

### Pushing to GitHub and HF from one working copy

```bash
./scripts/setup_remotes.sh <github-user>/<repo> <hf-user>/<space-name>
git add -A && git commit -m "initial commit"
git push origin main     # GitHub
git push space main      # Hugging Face
```

Or run `make push` after the remotes exist.

There's also `.github/workflows/sync-to-hf.yml`, which mirrors `main` to the
Space on every push once you add an `HF_TOKEN` secret (a write token from
https://huggingface.co/settings/tokens) and set `HF_SPACE` in the workflow env.

### Keeping the Space awake

Free Spaces sleep when idle, and a ping from inside the container won't stop
it — the request has to arrive through the HF ingress proxy. Set up a free
external monitor (UptimeRobot or similar) against
`https://<user>-<space>.hf.space/health` at a 30-minute interval.

## Notes and known limits

- **State is ephemeral.** SQLite lives in the container filesystem and is wiped
  on restart; the agent re-seeds itself and continues. For durability, point
  `DB_PATH` at a mounted volume or reimplement `db.py` against Turso/libSQL or
  Postgres — the rest of the app only touches the public methods on
  `Database`.
- **Single process only.** The llama.cpp context and the SQLite connection are
  per-process, hence `--workers 1`.
- **JSON mode over GBNF.** Qwen2.5-Instruct supports `response_format=
  {"type": "json_object"}` natively in llama-cpp-python, which is far more
  stable than hand-written GBNF grammars. The scout still falls back to a safe
  "not relevant" decision if parsing fails.
- **Scraping responsibly is on you.** `curl_cffi` impersonates a real browser
  fingerprint, which is there to stop ordinary sites from rejecting a legitimate
  client, not to defeat a site that has told you to go away. This matters more
  in open-web mode, where the agent touches thousands of hosts that never opted
  in: keep `RESPECT_ROBOTS=true`, keep `PER_DOMAIN_DELAY_SECONDS` at 15 or
  above, and keep the per-domain cap on. A polite crawler is also a crawler that
  doesn't get its Space IP-banned.

## License

MIT — see `LICENSE`.
