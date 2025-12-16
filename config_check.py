"""
Utility: validate project configuration and expected files.

This script is intended for quick sanity checks before running collection,
cleaning, and analysis steps. It prints a human-readable report and returns a
non-zero exit code if required configuration is missing.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv


@dataclass(frozen=True)
class CheckResult:
    """Result of a single check."""

    name: str
    ok: bool
    message: str


def _mask_secret(value: Optional[str]) -> str:
    """
    Mask secrets for printing.

    Args:
        value: Secret value.

    Returns:
        Masked string safe for logs.
    """
    if not value:
        return "(missing)"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def check_env(required: Iterable[str]) -> List[CheckResult]:
    """
    Check that required environment variables are set.

    Args:
        required: Names of required environment variables.

    Returns:
        List of check results.
    """
    results: List[CheckResult] = []
    for name in required:
        value = os.getenv(name)
        ok = bool(value)
        display = _mask_secret(value) if "PASSWORD" in name else (value or "(missing)")
        results.append(CheckResult(name=f"env:{name}", ok=ok, message=f"{name}={display}"))
    return results


def check_paths(paths: Iterable[Path]) -> List[CheckResult]:
    """
    Check that expected filesystem paths exist.

    Args:
        paths: Paths to check.

    Returns:
        List of check results.
    """
    results: List[CheckResult] = []
    for path in paths:
        if path.exists():
            results.append(CheckResult(name=f"path:{path}", ok=True, message="exists"))
        else:
            results.append(CheckResult(name=f"path:{path}", ok=False, message="missing"))
    return results


def print_report(results: List[CheckResult]) -> None:
    """
    Print a readable report of check results.

    Args:
        results: Check results to print.
    """
    width = max((len(r.name) for r in results), default=20)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"{status:<4} {r.name:<{width}}  {r.message}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate project configuration and expected files")
    parser.add_argument("--no-dotenv", action="store_true", help="Do not load variables from .env")
    parser.add_argument(
        "--mode",
        choices=["all", "env", "paths"],
        default="all",
        help="Which checks to run",
    )
    args = parser.parse_args()

    if not args.no_dotenv:
        load_dotenv()

    required_env = ["BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD"]
    expected_paths = [
        Path("data/raw/bluesky_posts.json"),
        Path("data/raw/manifest.json"),
        Path("data/processed/cleaned_posts.csv"),
    ]

    results: List[CheckResult] = []
    if args.mode in ("all", "env"):
        results.extend(check_env(required_env))
    if args.mode in ("all", "paths"):
        results.extend(check_paths(expected_paths))

    print_report(results)
    failed = [r for r in results if not r.ok]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

