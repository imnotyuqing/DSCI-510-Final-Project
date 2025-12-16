import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from langdetect import DetectorFactory, LangDetectException, detect

import pandas as pd

RAW_PATH = Path("data/raw/bluesky_posts.json")
PROCESSED_PATH = Path("data/processed/cleaned_posts.csv")

logger = logging.getLogger(__name__)

DetectorFactory.seed = 0
MIN_CHARS = 50


URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_.-]+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")
AD_HASHTAG_RE = re.compile(r"#(ad|sponsored|promo|giveaway)", re.IGNORECASE)

CONTEXT_BUCKETS = {
    "policy": ["policy", "guidelines", "compliance", "ferpa", "copyright", "ethics", "responsible ai"],
    "assessment": ["exam", "test", "quiz", "grading", "rubric", "proctoring", "cheating", "integrity"],
    "curriculum": ["syllabus", "course design", "lesson plan", "unit","module","assignment","homework",],
    "support_services": ["tutoring", "advising", "counseling", "library", "accessibility", "accommodations"],
    "edtech_tools": ["lms", "canvas", "blackboard", "moodle", "google classroom", "zoom", "teams"],
    "infrastructure": ["network", "devices", "labs", "deployment", "integration"],
    "professional_dev": ["pd", "training", "workshop", "faculty development", "teacher prep"],
    "stakeholders": ["parent", "parents", "caregiver", "caregivers", "community", "board", "donors"],
}

AD_KEYWORDS = [ "buy now", "limited time", "discount", "sale", "promo","promotion","deal","offer","use code",
    "subscribe","sign up","dm for","affiliate","sponsored","sponsor","find me at","copywriter","open to clients",
    "virtual assistant","author assistant",]

TRENDING_SPAM_PHRASES = ["bluesky's top 10 trending words", "top 10 trending words"]


def configure_logging() -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_raw_posts(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Load newline-delimited JSON records from disk.
    path: NDJSON file path.
    Returns: Parsed records, one JSON object per line.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def dedupe(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    De-duplicate records by `id` or `uri`.
    records: Iterable of raw post records.
    Returns: List of unique records, preserving first-seen order.
    """
    seen = set()
    unique: List[Dict[str, Any]] = []
    for rec in records:
        pid = rec.get("id") or rec.get("uri")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        unique.append(rec)
    return unique


def filter_min_length(records: List[Dict[str, Any]], min_chars: int) -> List[Dict[str, Any]]:
    """
    Filter out posts whose raw text is shorter than a threshold.
    records: List of post records with a `text` field.
    min_chars: Minimum required character count for raw `text`.
    Returns: Filtered list of records.
    """
    return [rec for rec in records if rec.get("text") and len(rec["text"]) >= min_chars]


def detect_language(text: str) -> str:
    """
    Detect the language code for a text string.
    text: Input text.
    Returns: Language code (e.g., "en") or "" if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return ""


def filter_language(records: List[Dict[str, Any]], target_lang: str = "en") -> List[Dict[str, Any]]:
    """
    Filter records by detected language.
    This function adds a `lang_detected` field to each record for transparency.
    records: List of post records.
    target_lang: ISO language code to keep (default: "en").
    Returns:. Filtered list of records.
    """
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        lang = detect_language(rec.get("text") or "")
        rec["lang_detected"] = lang
        if lang == target_lang:
            filtered.append(rec)
    return filtered


def remove_duplicate_text(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove posts that have the same cleaned text content.
    records: List of post records.
    Returns: Records with duplicate cleaned text removed. Adds a temporary
    `_text_clean_tmp` field used by later steps.
    """
    seen = set()
    unique: List[Dict[str, Any]] = []
    for rec in records:
        cleaned = clean_text(rec.get("text") or "")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        rec["_text_clean_tmp"] = cleaned
        unique.append(rec)
    return unique


def is_ad(rec: Mapping[str, Any]) -> bool:
    """
    Heuristic ad detection based on promo keywords, hashtags, and heavy URL usage.
    rec: Post record.
    Returns: True if the record is likely to be an ad/spam post.
    """
    raw_text = rec.get("text") or ""
    cleaned = rec.get("_text_clean_tmp") or clean_text(raw_text)
    lower_raw = raw_text.lower()

    has_keyword = any(k in cleaned for k in AD_KEYWORDS)
    has_hashtag = bool(AD_HASHTAG_RE.search(lower_raw))
    url_count = len(URL_RE.findall(raw_text))
    trending_spam = any(p in cleaned for p in TRENDING_SPAM_PHRASES)

    return has_keyword or has_hashtag or trending_spam or url_count >= 2


def filter_ads(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out likely ads/spam posts using `is_ad`.
    records: List of post records.
    Returns: Records that are not flagged as ads.
    """
    return [rec for rec in records if not is_ad(rec)]


def clean_text(text: str) -> str:
    """
    Normalize post text for de-duplication and feature extraction.
    Steps:
      - lowercase
      - remove URLs, mentions, emoji
      - normalize whitespace
    text: Raw post text.
    Returns: Cleaned text string.
    """
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

def tag_context(text: str) -> Dict[str, bool]:
    """
    Assign boolean education-context flags based on keyword presence.
    text: Cleaned text (typically output of `clean_text`).
    Returns: Mapping from context bucket name to True/False.
    """
    tags = {}
    for bucket, terms in CONTEXT_BUCKETS.items():
        tags[bucket] = any(term in text for term in terms)
    return tags

def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert cleaned records to a DataFrame with raw text, cleaned text, metadata, and context flags.
    records: List of post records.
    Returns: DataFrame ready for analysis.
    """
    rows = []
    for rec in records:
        raw_text = rec.get("text") or ""
        cleaned = rec.pop("_text_clean_tmp", None) or clean_text(raw_text)
        context_flags = tag_context(cleaned)
        row = {
            "id": rec.get("id"),
            "uri": rec.get("uri"),
            "query": rec.get("query"),
            "text": raw_text,
            "text_clean": cleaned,
            "created_at": rec.get("created_at"),
            "author_handle": rec.get("author_handle"),
            "author_display": rec.get("author_display"),
            "like_count": rec.get("like_count"),
            "repost_count": rec.get("repost_count"),
            "reply_count": rec.get("reply_count"),
            "lang": rec.get("lang"),
            "root": rec.get("root"),
            "parent": rec.get("parent"),
        }
        row.update(context_flags)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def quality_report(df: pd.DataFrame, dropped: Counter) -> None:
    """
    Log a small QA report describing how many rows were kept/dropped.
    df: Final cleaned DataFrame.
    dropped: Counts of records dropped at each stage.
    """
    logger.info("Rows kept: %d", len(df))
    logger.info("Dropped by language: %d", dropped.get("lang", 0))
    logger.info("Dropped by length: %d", dropped.get("len", 0))
    logger.info("Dropped by duplicate text: %d", dropped.get("dup_text", 0))
    logger.info("Dropped as ads: %d", dropped.get("ads", 0))
    if not df.empty:
        null_counts = df.isna().sum()
        logger.info("Null counts: %s", null_counts.to_dict())

def ensure_dirs() -> None:
    """Ensure output directories exist."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

def main() -> None:
    """CLI entrypoint for cleaning raw Bluesky posts into an analysis-ready CSV."""
    configure_logging()
    logger.info("Loading raw posts from %s", RAW_PATH)
    raw_records = list(load_raw_posts(RAW_PATH))
    logger.info("Raw count: %d", len(raw_records))

    deduped = dedupe(raw_records)
    dropped = Counter()
    lang_filtered = filter_language(deduped, target_lang="en")
    dropped["lang"] = len(deduped) - len(lang_filtered)
    length_filtered = filter_min_length(lang_filtered, min_chars=MIN_CHARS)
    dropped["len"] = len(lang_filtered) - len(length_filtered)
    text_deduped = remove_duplicate_text(length_filtered)
    dropped["dup_text"] = len(length_filtered) - len(text_deduped)
    ad_filtered = filter_ads(text_deduped)
    dropped["ads"] = len(text_deduped) - len(ad_filtered)

    df = records_to_dataframe(ad_filtered)

    ensure_dirs()
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved cleaned data to %s", PROCESSED_PATH)
    quality_report(df, dropped)

if __name__ == "__main__":
    main()
