"""
Utility: sample posts from the cleaned dataset for qualitative inspection.

This script creates a CSV of sampled posts to support manual validation of:
- context bucket tagging
- topic model interpretation (if topics are available)
- sentiment extremes (optional)
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


DEFAULT_BUCKETS: List[str] = [
    "policy",
    "assessment",
    "curriculum",
    "support_services",
    "edtech_tools",
    "infrastructure",
    "professional_dev",
    "stakeholders",
]


def load_cleaned(path: Path) -> pd.DataFrame:
    """
    Load the cleaned CSV.

    Args:
        path: Path to `cleaned_posts.csv`.

    Returns:
        Cleaned posts DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {path}")
    return pd.read_csv(path)


def sample_random(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample `n` rows uniformly at random.

    Args:
        df: Input DataFrame.
        n: Number of rows to sample.
        seed: RNG seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed)


def sample_by_bucket(df: pd.DataFrame, buckets: Sequence[str], per_bucket: int, seed: int) -> pd.DataFrame:
    """
    Sample posts stratified by context bucket.

    Args:
        df: Input DataFrame with boolean bucket columns.
        buckets: Bucket column names to sample from.
        per_bucket: Number of posts to sample per bucket.
        seed: RNG seed for reproducibility.

    Returns:
        Concatenated samples with a `sample_group` column indicating the bucket.
    """
    rng = random.Random(seed)
    parts: List[pd.DataFrame] = []
    for bucket in buckets:
        if bucket not in df.columns:
            continue
        subset = df[df[bucket].fillna(False).astype(bool)]
        if subset.empty:
            continue
        n = min(per_bucket, len(subset))
        # pandas sample uses numpy RNG; vary random_state per bucket deterministically
        random_state = rng.randint(0, 2**31 - 1)
        sampled = subset.sample(n=n, random_state=random_state).copy()
        sampled["sample_group"] = bucket
        parts.append(sampled)
    if not parts:
        return pd.DataFrame(columns=list(df.columns) + ["sample_group"])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Sample posts for qualitative inspection")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_posts.csv"),
        help="Path to cleaned CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/samples.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        choices=["random", "bucket"],
        default="bucket",
        help="Sampling mode",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of posts to sample in random mode")
    parser.add_argument("--per-bucket", type=int, default=30, help="Posts per bucket in bucket mode")
    parser.add_argument(
        "--buckets",
        type=str,
        default=",".join(DEFAULT_BUCKETS),
        help="Comma-separated bucket columns to sample",
    )
    args = parser.parse_args()

    df = load_cleaned(args.input)

    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    if args.mode == "random":
        out = sample_random(df, n=args.n, seed=args.seed).copy()
        out["sample_group"] = "random"
    else:
        out = sample_by_bucket(df, buckets=buckets, per_bucket=args.per_bucket, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        c
        for c in [
            "sample_group",
            "created_at",
            "author_handle",
            "query",
            "text",
            "text_clean",
            *buckets,
        ]
        if c in out.columns
    ]
    out = out[keep_cols]
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} sampled posts to {args.output}")


if __name__ == "__main__":
    main()

