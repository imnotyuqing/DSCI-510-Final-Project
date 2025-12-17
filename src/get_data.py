import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

BASE_URL = "https://bsky.social"
SEARCH_ENDPOINT = f"{BASE_URL}/xrpc/app.bsky.feed.searchPosts"
SESSION_ENDPOINT = f"{BASE_URL}/xrpc/com.atproto.server.createSession"

RAW_PATH = Path("data/raw/bluesky_posts.json")
MANIFEST_PATH = Path("data/raw/manifest.json")

DEFAULT_AI_TERMS = [
    "chatgpt",
    "gpt",
    "gemini",
    "claude",
    "llm",
    "openai",
    "anthropic",
    "ai",
    "gen ai",
]

DEFAULT_EDU_TERMS = [
    "student",
    "teacher",
    "faculty",
    "professor",
    "admin",
    "principal",
    "superintendent",
    "school",
    "classroom",
    "syllabus",
    "assignment",
    "grading",
]

logger = logging.getLogger(__name__)

@dataclass
class Config:
    handle: str
    app_password: str
    start_date: datetime
    end_date: datetime
    ai_terms: List[str]
    edu_terms: List[str]
    limit_per_query: int = 100

def parse_date(value: str) -> datetime:
    """Parse an ISO date string and return a timezone-aware UTC datetime."""
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

def split_terms(raw: Optional[str], default: Sequence[str]) -> List[str]:
    """
    Split a comma-separated environment variable into a list of non-empty terms.
    raw: Comma-separated string from an environment variable (or None).
    default: Default terms to use when `raw` is missing/empty.
    Returns a list of cleaned terms.
    """
    if not raw:
        return list(default)
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def load_config() -> Config:
    """
    Load credentials and query configuration from environment variables (and .env).
    Required environment variables:
      - BLUESKY_HANDLE
      - BLUESKY_APP_PASSWORD
    Returns a validated `Config` instance.
    """
    load_dotenv()
    handle = os.getenv("BLUESKY_HANDLE")
    app_password = os.getenv("BLUESKY_APP_PASSWORD")
    start_date_raw = os.getenv("START_DATE")
    end_date_raw = os.getenv("END_DATE")
    ai_terms_raw = os.getenv("AI_TERMS")
    edu_terms_raw = os.getenv("EDU_TERMS")

    missing = []
    names = [("BLUESKY_HANDLE", handle), ("BLUESKY_APP_PASSWORD", app_password)]
    for name, val in names:
        if not val:
            missing.append(name)
    
    if missing:
        raise ValueError(f"Missing required environment variables.")

    start_date = parse_date(start_date_raw) if start_date_raw else parse_date("2024-01-01")
    end_date = parse_date(end_date_raw) if end_date_raw else datetime.now(tz=timezone.utc)

    ai_terms = split_terms(ai_terms_raw, DEFAULT_AI_TERMS)
    edu_terms = split_terms(edu_terms_raw, DEFAULT_EDU_TERMS)

    if not ai_terms or not edu_terms:
        raise ValueError("AI_TERMS and EDU_TERMS must not be empty.")

    return Config(
        handle=handle,
        app_password=app_password,
        start_date=start_date,
        end_date=end_date,
        ai_terms=ai_terms,
        edu_terms=edu_terms,
    )

def build_queries(ai_terms: Sequence[str], edu_terms: Sequence[str]) -> List[str]:
    """
    Build deduplicated search queries from AI terms and education terms.
    Each query is constructed as `"AI_TERM EDU_TERM"`. Queries are deduped and sorted to ensure stable iteration order.
    ai_terms: AI-related keywords (e.g., "chatgpt", "llm").
    edu_terms: Education-related keywords (e.g., "teacher", "assignment").
    Returns sorted list of query strings.
    """    
    queries = {f"{ai} {edu}" for ai in ai_terms for edu in edu_terms}
    return sorted(queries)

def create_session(handle: str, app_password: str) -> str:
    """
    Authenticate to Bluesky and return an access JWT.
    handle: Bluesky handle (identifier).
    app_password: Bluesky app password.
    Returns access JWT used for subsequent API requests.
    """
    resp = requests.post(
        SESSION_ENDPOINT,
        json={"identifier": handle, "password": app_password},
        timeout=15,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to create session: {resp.status_code} {resp.text}")
    
    data = resp.json()
    jwt = data.get("accessJwt")

    if not jwt:
        raise RuntimeError("No accessJwt in session response.")
    return jwt

def fetch_page(query: str, jwt: str, cursor: Optional[str] = None, limit: int = 100, max_retries: int = 3,) -> Dict[str, Any]:
    """
    Fetch a page of search results for a query from the Bluesky searchPosts endpoint.
    query: Search query string.
    jwt: Access JWT from `create_session`.
    cursor: Pagination cursor returned by prior requests (or None).
    limit: Per-request page size.
    max_retries: Retry count for 429/5xx responses.
    Returns parsed JSON response.
    """
    headers = {"Authorization": f"Bearer {jwt}"}
    params = {"q": query, "limit": limit}
    if cursor:
        params["cursor"] = cursor

    for attempt in range(1, max_retries + 1):
        resp = requests.get(SEARCH_ENDPOINT, headers=headers, params=params, timeout=20)

        if resp.status_code == 429:
            sleep_for = 5 * attempt
            logger.warning("Rate limited. Sleeping for %s seconds.", sleep_for)
            time.sleep(sleep_for)
            continue

        if resp.status_code >= 500:
            sleep_for = 2 * attempt
            logger.warning(
                "Upstream error (%s). Attempt %s/%s. Sleeping for %s seconds.",
                resp.status_code,
                attempt,
                max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)
            continue
       
        if resp.status_code != 200:
            raise RuntimeError(f"Search failed ({resp.status_code}): {resp.text}")
        return resp.json()

    raise RuntimeError(f"Search failed after {max_retries} retries for query '{query}': {resp.text}")

def created_at_in_window(post: Mapping[str, Any], start: datetime, end: datetime) -> bool:
    """
    Check whether a Bluesky post's `createdAt` is within a target time window.
    post: Raw post object returned by the Bluesky API.
    start: Inclusive window start (UTC).
    end: Inclusive window end (UTC).
    Returns True if the post has a parsable `createdAt` within `[start, end]`.
    """
    created_str = post.get("record", {}).get("createdAt")
    if not created_str:
        return False
    try:
        created = datetime.fromisoformat(created_str.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        return False
    return start <= created <= end
def normalize_post(post: Mapping[str, Any], query: str) -> Dict[str, Any]:
    """
    Normalize a raw Bluesky API post into a flat record for storage/analysis.
    post: Raw post object returned by the Bluesky API.
    query: Query string responsible for retrieving this post.
    Returns flattened dictionary containing fields used downstream.
    """
    record = post.get("record", {})
    author = post.get("author", {})
    return {
        "id": post.get("uri"),
        "cid": post.get("cid"),
        "query": query,
        "text": record.get("text"),
        "created_at": record.get("createdAt"),
        "author_handle": author.get("handle"),
        "author_display": author.get("displayName"),
        "like_count": post.get("likeCount"),
        "repost_count": post.get("repostCount"),
        "reply_count": post.get("replyCount"),
        "lang": record.get("lang"),
        "labels": post.get("labels", []),
        "uri": post.get("uri"),
        "root": record.get("reply", {}).get("root", {}).get("uri") if record.get("reply") else None,
        "parent": record.get("reply", {}).get("parent", {}).get("uri") if record.get("reply") else None,
    }
def collect_posts(config: Config) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect posts across AI×education queries within the configured time window.
    The function de-duplicates globally across all queries by post URI and returns
    both the collected records and a manifest summarizing the run.
    config: Collection configuration.
    Returns a tuple of (posts, manifest).
    """
    jwt = create_session(config.handle, config.app_password)
    queries = build_queries(config.ai_terms, config.edu_terms)
    logger.info("Built %d queries.", len(queries))

    seen_ids = set()
    collected: List[Dict[str, Any]] = []
    total_api_calls = 0

    for query in tqdm(queries, desc="Queries"):
        cursor = None
        while True:
            data = fetch_page(query, jwt, cursor=cursor, limit=config.limit_per_query)
            total_api_calls += 1
            posts = data.get("posts", [])
            for raw in posts:
                if not created_at_in_window(raw, config.start_date, config.end_date):
                    continue
                norm = normalize_post(raw, query)
                pid = norm.get("id")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                collected.append(norm)
            cursor = data.get("cursor")
            if not cursor or not posts:
                break
            time.sleep(0.25)  # Be polite to API.

    manifest = {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "ai_terms": config.ai_terms,
        "edu_terms": config.edu_terms,
        "query_count": len(queries),
        "post_count": len(collected),
        "api_calls": total_api_calls,
    }
    return collected, manifest

def ensure_dirs() -> None:
    """Ensure output directories exist."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

def write_ndjson(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """
    Write records as newline-delimited JSON (NDJSON).
    records: Iterable of JSON-serializable mappings.
    path: Output file path.
    """
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    """
    Write a JSON manifest describing the collection run.
    manifest: Manifest dictionary.
    path: Output file path.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def configure_logging() -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def main() -> None:
    """CLI entrypoint for collecting raw Bluesky posts."""
    configure_logging()
    try:
        config = load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        raise

    logger.info(
        "Collecting posts for %d AI terms × %d edu terms between %s and %s",
        len(config.ai_terms),
        len(config.edu_terms),
        config.start_date.date(),
        config.end_date.date(),
    )
    ensure_dirs()
    posts, manifest = collect_posts(config)
    write_ndjson(posts, RAW_PATH)
    write_manifest(manifest, MANIFEST_PATH)
    logger.info("Saved %d posts to %s", len(posts), RAW_PATH)
    logger.info("Manifest saved to %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
