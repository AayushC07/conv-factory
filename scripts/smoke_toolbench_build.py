#!/usr/bin/env python3
"""Smoke-test `convfactory build` on a ToolBench directory (optionally cap tool count).

Use this after downloading ToolBench data to verify the loader and graph build on a
large slice without running generation.

Example::

    python scripts/smoke_toolbench_build.py --data-dir /path/to/toolbench --max-tools 200
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow running without installing the package as a script module
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conversation_factory.pipeline import ConversationPipeline  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Build registry + tool graph (smoke test).")
    p.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        required=True,
        help="ToolBench tools directory",
    )
    p.add_argument(
        "--max-tools",
        "-m",
        type=int,
        default=None,
        help="Maximum tools to load (default: all)",
    )
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = ConversationPipeline(
            data_dir=args.data_dir,
            output_dir=Path(tmp),
            max_tools=args.max_tools,
            seed=42,
        )
        summary = pipeline.build()

    print(json.dumps(summary, indent=2, default=str))
    reg = summary.get("registry", {})
    g = summary.get("graph", {})
    print(
        f"OK — loaded {reg.get('num_tools', 0)} tools, "
        f"{g.get('num_nodes', 0)} graph nodes, {g.get('num_edges', 0)} edges.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
