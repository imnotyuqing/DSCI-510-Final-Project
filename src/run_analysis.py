import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_PATH = Path("data/processed/cleaned_posts.csv")
CONTEXT_BUCKETS: List[str] = [
    "policy",
    "assessment",
    "curriculum",
    "support_services",
    "edtech_tools",
    "infrastructure",
    "professional_dev",
    "stakeholders",
]
KEY_FIELDS = ["text", "text_clean", "created_at", "author_handle"]


def configure_logging(verbose: bool) -> None:
    """
    Configure logging for CLI execution.
    verbose: If True, use DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def load_cleaned(path: Path) -> pd.DataFrame:
    """
    Load the cleaned CSV into a DataFrame.
    path: Path to the cleaned CSV.
    Returns: Loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_csv(path)
    logging.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def report_nulls(df: pd.DataFrame) -> None:
    """
    Log null counts for key fields and the number of rows missing both text fields.
    df: Cleaned posts DataFrame.
    """
    logging.info("Null counts for key fields:")
    for field in KEY_FIELDS:
        count = int(df[field].isna().sum())
        logging.info("  %s: %d", field, count)
    missing_both = int((df["text"].isna() & df["text_clean"].isna()).sum())
    logging.info("Rows missing both text and text_clean: %d", missing_both)


def report_lengths(df: pd.DataFrame) -> None:
    """
    Log descriptive statistics of the cleaned text length distribution.

    Args:
        df: Cleaned posts DataFrame.
    """
    lengths = df["text_clean"].fillna("").str.len()
    percentiles = lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    logging.info("text_clean length stats (chars):")
    for key in ["count", "mean", "std", "min", "50%", "90%", "95%", "99%", "max"]:
        logging.info("  %s: %s", key, percentiles.get(key))
    over_500 = int((lengths > 500).sum())
    logging.info("  >500 chars: %d", over_500)


def report_context_buckets(df: pd.DataFrame) -> None:
    """
    Print a context bucket prevalence table and log per-bucket counts/percentages.
    df: Cleaned posts DataFrame containing boolean bucket columns.
    """
    logging.info("Context bucket prevalence:")
    rows = []
    for bucket in CONTEXT_BUCKETS:
        if bucket not in df.columns:
            logging.warning("  %s: missing column", bucket)
            continue
        count = int(df[bucket].sum())
        pct = df[bucket].mean() * 100
        logging.info("  %s: count=%d, pct=%.2f%%", bucket, count, pct)
        rows.append({"bucket": bucket, "count": count, "pct": pct})

    if rows:
        table = pd.DataFrame(rows)
        table = table.sort_values(by="pct", ascending=False).reset_index(drop=True)
        print("\nContext prevalence table (sorted by pct):")
        print(table.to_string(index=False, formatters={"pct": "{:.2f}".format}))


def get_vectorizer(max_features: int = 50000) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer configured for unigrams and bigrams.
    max_features: Maximum size of the vocabulary.
    Returns: Configured `TfidfVectorizer` instance.
    """
    # Use English stopwords and include unigrams + bigrams to surface phrases.
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=5,
    )


def top_terms(
    tfidf_matrix: spmatrix,
    feature_names: Sequence[str],
    top_n: int = 15,
) -> List[Tuple[str, float]]:
    """
    Compute top terms by aggregate TF-IDF weight across a set of documents.
    tfidf_matrix: Sparse document-term TF-IDF matrix.
    feature_names: Vocabulary terms aligned to matrix columns.
    top_n: Number of terms to return.
    Returns: List of (term, score) pairs ordered by descending score.
    """
    scores = tfidf_matrix.sum(axis=0).A1
    if len(scores) == 0:
        return []
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(str(feature_names[i]), float(scores[i])) for i in top_idx]

def report_top_terms(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Print top TF-IDF terms overall and by context bucket.
    df: Cleaned posts DataFrame with `text_clean` and bucket columns.
    top_n: Number of terms/phrases to print per group.
    """
    texts = df["text_clean"].fillna("")
    vectorizer = get_vectorizer()
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    overall_top = top_terms(tfidf, feature_names, top_n=top_n)
    print("\nTop tf-idf terms (overall):")
    for term, score in overall_top:
        print(f"  {term:<25} {score:.4f}")

    print("\nTop tf-idf terms by context bucket:")
    for bucket in CONTEXT_BUCKETS:
        if bucket not in df.columns:
            continue
        mask = df[bucket].fillna(False).astype(bool).values
        if mask.sum() == 0:
            print(f"  {bucket}: no posts")
            continue
        tfidf_bucket = tfidf[mask]
        bucket_top = top_terms(tfidf_bucket, feature_names, top_n=top_n)
        print(f"  {bucket}:")
        for term, score in bucket_top:
            print(f"    {term:<25} {score:.4f}")

def main() -> None:
    """CLI entrypoint for baseline analysis and top-terms reporting."""
    parser = argparse.ArgumentParser(description="Baseline analysis on cleaned Bluesky posts")
    parser.add_argument("--path", type=Path, default=PROCESSED_PATH, help="Path to cleaned CSV")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top tf-idf terms to show")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    df = load_cleaned(args.path)
    report_nulls(df)
    report_lengths(df)
    report_context_buckets(df)
    report_top_terms(df, top_n=args.top_n)


if __name__ == "__main__":
    main()
