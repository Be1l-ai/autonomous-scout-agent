"""Central configuration. Every field can be overridden by an env var of the
same name (upper-cased) or by a line in .env — that is how HF Space secrets
and GitHub Actions secrets get in."""

from __future__ import annotations

import glob
import os
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hosts that eat a crawler's time without ever containing what you asked for.
DEFAULT_BLOCKLIST = (
    "doubleclick.net,googleadservices.com,googlesyndication.com,google-analytics.com,"
    "facebook.com,instagram.com,twitter.com,x.com,tiktok.com,pinterest.com,"
    "linkedin.com,accounts.google.com,login.microsoftonline.com,"
    "amazon.com,ebay.com,aliexpress.com,"
    "youtube.com,netflix.com,spotify.com"
)

# Priority hints: URLs containing these paths get priority 1 (highest)
LIST_HINTS = (
    "/list",
    "/awesome",
    "/awesome-",
    "/resources",
    "/tools",
    "/projects",
    "/repositories",
    "/awesome-list",
    "/curated",
    "/index",
    "/catalog",
    "/directory",
    "/landscape",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Scout model (local GGUF) -------------------------------------------
    scout_repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    scout_quant: str = "q4_k_m"
    scout_model_dir: str = "./models"
    scout_model_path: str = ""      # empty -> auto-discover inside scout_model_dir
    scout_context_size: int = 4096
    scout_threads: int = 2          # keep low: HF free tier gives 2 vCPU
    scout_temperature: float = 0.1
    scout_max_tokens: int = 512
    # Below this confidence, a "relevant" verdict won't spend a worker call.
    scout_min_confidence: float = 0.4

    # --- Worker (Groq) ------------------------------------------------------
    worker_provider: str = "groq"
    worker_api_key: str = Field(default="")
    worker_model: str = "groq/compound-mini"
    worker_max_tokens: int = 1024

    # Optional GitHub token (no scopes needed) — raises the Search API limit
    # from 10 to 30 req/min when resolving repo URLs for extracted items.
    github_token: str = ""

    # --- Discovery / search -------------------------------------------------
    # How the agent finds NEW starting points instead of only following links.
    # One of: none | brave | tavily | searxng
    search_provider: str = "none"
    search_api_key: str = Field(default="")
    searxng_url: str = "https://searxng.example.com"
    # Comma-separated. Leave empty to have the scout derive queries from the goal.
    search_queries: str = ""
    search_results_per_query: int = 10
    # Re-run discovery when the queue runs dry.
    search_on_empty_queue: bool = True

    # --- Storage ------------------------------------------------------------
    db_path: str = "./data/agent_state.db"

    # --- Fetcher ------------------------------------------------------------
    fetch_timeout: int = 20
    impersonate: str = "chrome124"
    respect_robots: bool = True
    max_page_chars: int = 6000
    max_links_per_page: int = 20
    max_content_bytes: int = 3_000_000   # skip huge pages/binaries

    # --- Crawl scope --------------------------------------------------------
    seed_urls: str = ""
    # EMPTY = the open web. Set host suffixes to fence the agent in.
    allowed_domains: str = ""
    blocked_domains: str = DEFAULT_BLOCKLIST
    # Hops from a seed. The web branches ~20x per page, so this matters a lot.
    max_depth: int = 3
    max_queue_size: int = 500
    max_pages: int = 0                   # 0 = unlimited
    # Stops the agent sinking its whole budget into one chatty site.
    max_pages_per_domain: int = 25

    # --- Politeness ---------------------------------------------------------
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 5.0
    # Minimum gap between two hits on the SAME host, independent of the above.
    per_domain_delay_seconds: float = 15.0
    idle_sleep_min: float = 10.0
    idle_sleep_max: float = 30.0

    # --- Agent goal ---------------------------------------------------------
    current_goal: str = "Find open-source AI agent frameworks and their source repositories."

    # --- Control plane auth -------------------------------------------------
    # Empty = disabled (dev mode). Set in production to secure /pause, /seed, etc.
    control_token: str = ""

    # --- Runtime ------------------------------------------------------------
    autostart: bool = True
    log_level: str = "info"

    # --- Derived helpers ----------------------------------------------------
    @staticmethod
    def _csv(value: str) -> List[str]:
        return [v.strip() for v in value.split(",") if v.strip()]

    @property
    def seed_url_list(self) -> List[str]:
        return self._csv(self.seed_urls)

    @property
    def allowed_domain_list(self) -> List[str]:
        return [d.lower() for d in self._csv(self.allowed_domains)]

    @property
    def blocked_domain_list(self) -> List[str]:
        return [d.lower() for d in self._csv(self.blocked_domains)]

    @property
    def search_query_list(self) -> List[str]:
        return self._csv(self.search_queries)

    @property
    def open_web(self) -> bool:
        return not self.allowed_domain_list

    def resolve_model_path(self) -> str:
        """Return an actual on-disk path to the GGUF weights.

        The quant filename in a HF GGUF repo is not always predictable (some are
        sharded, some are capitalised differently), so we glob rather than
        hard-code a filename.
        """
        if self.scout_model_path and os.path.exists(self.scout_model_path):
            return self.scout_model_path

        patterns = [
            os.path.join(self.scout_model_dir, f"*{self.scout_quant}*.gguf"),
            os.path.join(self.scout_model_dir, "**", f"*{self.scout_quant}*.gguf"),
            os.path.join(self.scout_model_dir, "*.gguf"),
            os.path.join(self.scout_model_dir, "**", "*.gguf"),
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True))
            if matches:
                return matches[0]   # first shard of a split model

        raise FileNotFoundError(
            f"No .gguf found in {self.scout_model_dir!r}. "
            "Run `python download_model.py` first."
        )


settings = Settings()
