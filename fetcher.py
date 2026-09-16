"""HTTP fetching + HTML -> clean text/links.

curl_cffi impersonates a real browser TLS/JA3 fingerprint so that ordinary
sites don't reject the client outright. We still honour robots.txt by default
(`RESPECT_ROBOTS=true`) and enforce a per-host cooldown — on the open web the
agent touches thousands of hosts it has no relationship with, so both matter
more here than in a single-site crawl.
"""

from __future__ import annotations

import threading
import time
import urllib.robotparser as robotparser
from typing import Dict, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import structlog
from bs4 import BeautifulSoup
from curl_cffi import requests
from curl_cffi.requests.errors import Timeout, ConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import trafilatura

from config import settings
from schemas import FetchResult

logger = structlog.get_logger()

NOISE_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")

# Extensions that are never worth fetching for a text-extraction agent.
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp4", ".webm", ".mov", ".avi", ".mp3", ".wav", ".ogg", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".dmg", ".iso",
    ".exe", ".msi", ".deb", ".rpm", ".apk",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
)

TEXTUAL_CONTENT_TYPES = ("text/html", "application/xhtml", "text/plain", "application/xml")


class RobotsCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._lock = threading.Lock()

    def allowed(self, url: str, user_agent: str = "*") -> bool:
        if not settings.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        with self._lock:
            if origin not in self._cache:
                self._cache[origin] = self._load(origin)
            rp = self._cache[origin]
        if rp is None:
            return True  # couldn't read robots.txt -> don't hard-block
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True

    @staticmethod
    def _load(origin: str) -> Optional[robotparser.RobotFileParser]:
        try:
            resp = requests.get(
                urljoin(origin, "/robots.txt"),
                timeout=10,
                impersonate=settings.impersonate,
            )
            if resp.status_code >= 400:
                return None
            rp = robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            return rp
        except Exception as exc:
            logger.debug("robots_fetch_failed", origin=origin, error=str(exc))
            return None


class DomainThrottle:
    """Per-host cooldown. The global jitter in the orchestrator doesn't stop the
    agent hammering one host if the queue happens to be full of its URLs."""

    def __init__(self) -> None:
        self._last_hit: Dict[str, float] = {}
        self._lock = threading.Lock()

    def ready(self, host: str) -> bool:
        with self._lock:
            last = self._last_hit.get(host, 0.0)
        return (time.time() - last) >= settings.per_domain_delay_seconds

    def mark(self, host: str) -> None:
        with self._lock:
            self._last_hit[host] = time.time()


class NonRetryableHTTPError(Exception):
    pass


class RateLimitedError(Exception):
    """Raised when the server returns 429 Too Many Requests."""

    def __init__(self, retry_after: int):
        # Default 30 min, max 1 hour
        self.retry_after = min(max(retry_after, 1), 3600)
        super().__init__(f"Rate limited, retry after {self.retry_after}s")


class Fetcher:
    def __init__(self) -> None:
        self.session = requests.Session(impersonate=settings.impersonate)
        self.robots = RobotsCache()
        self.throttle = DomainThrottle()

    @staticmethod
    def host_of(url: str) -> str:
        return (urlparse(url).hostname or "").lower()

    @staticmethod
    def is_fetchable(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        return not parsed.path.lower().endswith(SKIP_EXTENSIONS)

    @staticmethod
    def _is_github_repo_url(url: str) -> bool:
        """Check if URL is a GitHub repo (owner/repo) that we can fetch via API."""
        parsed = urlparse(url)
        if parsed.netloc.lower() != "github.com":
            return False
        path = parsed.path.strip("/")
        # Exclude special paths
        if path.startswith(("topics/", "search", "orgs/")):
            return False
        # Repo path should be exactly "owner/repo" (2 segments)
        segments = [s for s in path.split("/") if s]
        return len(segments) == 2

    def _fetch_github_repo(self, url: str) -> Optional[FetchResult]:
        """Fetch GitHub repo data via API. Returns None on failure (fallback to HTML)."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        api_url = f"https://api.github.com/repos/{path}"
        headers = {"Accept": "application/vnd.github+json"}
        try:
            resp = self.session.get(api_url, headers=headers, timeout=settings.fetch_timeout)
            if resp.status_code == 403:
                # Rate limited - no Retry-After header on GitHub API typically
                logger.warning("github_api_rate_limited", url=url)
                return None
            if resp.status_code != 200:
                return None
            data = resp.json()
            # Build rich text from API response
            text_parts = [
                f"Repository: {data.get('full_name', '')}",
                f"Description: {data.get('description', 'No description')}",
                f"Stars: {data.get('stargazers_count', 0)}",
                f"Language: {data.get('language', 'Unknown')}",
                f"Topics: {', '.join(data.get('topics', []))}",
                f"License: {data.get('license', {}).get('name', 'None') if data.get('license') else 'None'}",
                f"URL: {data.get('html_url', '')}",
            ]
            text = "\n".join(text_parts)
            return FetchResult(
                url=url,
                status_code=200,
                text=text,
                links=[data.get("html_url", "")] if data.get("html_url") else [],
                title=data.get("full_name", ""),
            )
        except Exception as exc:
            logger.debug("github_api_fetch_failed", url=url, error=str(exc))
            return None

    def is_allowed(self, url: str) -> bool:
        return self.robots.allowed(url)

    def ready_for(self, url: str) -> bool:
        return self.throttle.ready(self.host_of(url))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((Timeout, ConnectionError))
    )
    def fetch_and_parse(self, url: str) -> FetchResult:
        logger.info("fetching_url", url=url)
        self.throttle.mark(self.host_of(url))

        # GitHub repo optimization: try API first for owner/repo URLs
        if self._is_github_repo_url(url):
            api_result = self._fetch_github_repo(url)
            if api_result:
                logger.info("github_api_success", url=url)
                return api_result
            logger.info("github_api_fallback", url=url)

        if "medium.com" in url or "pub.towardsai.net" in url:
            raise NonRetryableHTTPError("Medium domains are blocked by default")

        response = self.session.get(url, timeout=settings.fetch_timeout)

        # Check for 429 BEFORE the 4xx check
        if response.status_code == 429:
            retry_after = 1800  # default 30 minutes
            retry_header = response.headers.get("Retry-After")
            if retry_header:
                try:
                    retry_after = int(retry_header)
                except ValueError:
                    pass
            raise RateLimitedError(retry_after)

        if response.status_code in [401, 403, 404]:
            raise NonRetryableHTTPError(f"Non-retryable HTTP {response.status_code}")

        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        if content_type and not any(t in content_type for t in TEXTUAL_CONTENT_TYPES):
            raise ValueError(f"non-textual content-type: {content_type}")
        if len(response.content) > settings.max_content_bytes:
            raise ValueError(f"page too large: {len(response.content)} bytes")

        # 1. Extract clean main content (ignores nav, footers, ads)
        text = trafilatura.extract(response.content, include_comments=False, include_tables=True)

        # 2. Fallback to BS4 if trafilatura fails
        if not text:
            soup = BeautifulSoup(response.content, "lxml")
            for tag in soup(list(NOISE_TAGS)):
                tag.extract()
            text = soup.get_text(separator=" ", strip=True)

        # 3. Increase limit since boilerplate is removed
        text = text[:12000]

        soup = BeautifulSoup(response.content, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""

        links = []
        seen = set()
        for anchor in soup.find_all("a", href=True):
            href, _ = urldefrag(urljoin(url, anchor["href"]))
            if href in seen or not self.is_fetchable(href):
                continue
            seen.add(href)
            links.append(href)
            if len(links) >= settings.max_links_per_page * 5:
                break

        return FetchResult(
            url=url,
            status_code=response.status_code,
            text=text,
            links=links,
            title=title,
        )
