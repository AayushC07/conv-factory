#!/usr/bin/env python3
"""Build a small ToolBench-style ``sample_data/`` tree from the full ``tools/`` corpus.

Copies a deterministic subset (default ~24 tools across 6 domains) so demos and
CI can use ``convfactory build --data-dir sample_data`` without loading tens of
thousands of tools.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# Prefer these domains when present under ``--source`` (diverse, common in demos).
DEFAULT_DOMAIN_PRIORITY: tuple[str, ...] = (
    "Travel",
    "Weather",
    "Food",
    "Finance",
    "Entertainment",
    "Commerce",
)


def _discover_domains(source: Path) -> list[str]:
    names: list[str] = []
    for p in sorted(source.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if any(p.glob("*.json")):
            names.append(p.name)
    return names


def _choose_domains(
    available: list[str],
    num_domains: int,
    priority: tuple[str, ...],
) -> list[str]:
    chosen: list[str] = []
    for name in priority:
        if name in available and name not in chosen:
            chosen.append(name)
        if len(chosen) >= num_domains:
            return chosen
    for name in available:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= num_domains:
            break
    if len(chosen) < num_domains:
        raise ValueError(
            f"Need {num_domains} domains but only found {len(chosen)} with *.json under source."
        )
    return chosen


def _split_counts(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _sample_files(
    domain_dir: Path,
    k: int,
    rng: random.Random,
) -> list[Path]:
    files = sorted(domain_dir.glob("*.json"))
    if not files:
        return []
    if k >= len(files):
        return list(files)
    return rng.sample(files, k)


def create_sample_data(
    *,
    source_dir: Path,
    output_dir: Path,
    total_tools: int,
    num_domains: int,
    seed: int,
    domains_override: list[str] | None,
    clean: bool,
) -> list[tuple[str, Path]]:
    """Copy selected tool JSON files into ``output_dir`` / category / file.

    Returns list of (category_name, dest_path) for logging.
    """
    available = _discover_domains(source_dir)
    if not available:
        raise FileNotFoundError(f"No domain subdirectories with *.json under {source_dir}")

    if domains_override:
        missing = [d for d in domains_override if d not in available]
        if missing:
            raise ValueError(f"Unknown or empty domains (no *.json): {missing}")
        chosen = domains_override[:]
        if len(chosen) != num_domains:
            raise ValueError(
                f"--domains has {len(chosen)} entries but --num-domains is {num_domains}"
            )
    else:
        chosen = _choose_domains(available, num_domains, DEFAULT_DOMAIN_PRIORITY)

    if total_tools < num_domains:
        raise ValueError("total_tools must be >= num_domains so each domain gets at least one tool.")

    rng = random.Random(seed)
    counts = _split_counts(total_tools, len(chosen))

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    copied: list[tuple[str, Path]] = []
    for domain, k in zip(chosen, counts, strict=True):
        ddir = source_dir / domain
        picks = _sample_files(ddir, k, rng)
        if len(picks) < k:
            raise RuntimeError(
                f"Domain {domain!r} has only {len(picks)} JSON tools; need {k}."
            )
        out_cat = output_dir / domain
        out_cat.mkdir(parents=True, exist_ok=True)
        for src in picks:
            dest = out_cat / src.name
            shutil.copy2(src, dest)
            copied.append((domain, dest))

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy 20–30 ToolBench tools from tools/ into sample_data/ for demos."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("tools"),
        help="Root directory containing domain folders with tool *.json files (default: tools)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_data"),
        help="Output directory (default: sample_data)",
    )
    parser.add_argument(
        "--tools",
        type=int,
        default=24,
        help="Total number of tools to copy (default: 24; use 20–30)",
    )
    parser.add_argument(
        "--num-domains",
        type=int,
        default=6,
        help="Number of domain folders to include (default: 6; use 5 or 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="",
        help="Comma-separated domain folder names (overrides auto-pick); must match --num-domains count",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing files under output before copying",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        print(f"error: source directory not found: {source}", file=sys.stderr)
        return 1

    override: list[str] | None = None
    if args.domains.strip():
        override = [x.strip() for x in args.domains.split(",") if x.strip()]

    try:
        copied = create_sample_data(
            source_dir=source,
            output_dir=args.output.resolve(),
            total_tools=args.tools,
            num_domains=args.num_domains,
            seed=args.seed,
            domains_override=override,
            clean=not args.no_clean,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    by_cat: dict[str, int] = {}
    for cat, _ in copied:
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print(f"Copied {len(copied)} tools into {args.output}/")
    for cat in sorted(by_cat):
        print(f"  {cat}: {by_cat[cat]}")
    print("\nRun:")
    print(f"  convfactory build --data-dir {args.output} --output-dir output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
