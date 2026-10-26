"""Export sampled tool chains to JSON and a simple HTML visualization."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .sampler import SamplingConstraints, ToolChain


def constraints_to_dict(c: SamplingConstraints) -> dict[str, Any]:
    d = asdict(c)
    d["excluded_endpoints"] = sorted(d["excluded_endpoints"])
    return d


def chains_document(
    chains: list[ToolChain],
    *,
    seed: int,
    constraints: SamplingConstraints,
    graph_path: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable document of sampled chains."""
    return {
        "seed": seed,
        "graph_path": graph_path,
        "constraints": constraints_to_dict(constraints),
        "num_chains": len(chains),
        "chains": [
            {
                "index": i,
                "endpoints": ch.endpoints,
                "pattern": ch.pattern,
                "metadata": ch.metadata,
                "num_steps": ch.num_steps,
                "distinct_tools": sorted(ch.distinct_tools),
                "is_multi_tool": ch.is_multi_tool,
            }
            for i, ch in enumerate(chains)
        ],
    }


def save_chains_json(path: str | Path, doc: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def write_chains_html(path: str | Path, doc: dict[str, Any], *, title: str = "Sampled tool chains") -> None:
    """Write a self-contained HTML page: each chain as a left-to-right sequence of endpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    chains = doc.get("chains", [])
    seed = doc.get("seed", "")
    constraints = doc.get("constraints", {})

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        f"<meta charset=\"utf-8\"><title>{html.escape(title)}</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #1a1a1a; }",
        "h1 { font-size: 1.25rem; }",
        "pre.meta { background: #eee; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; overflow-x: auto; }",
        ".chain { margin-bottom: 1.75rem; padding: 1rem; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }",
        ".chain h2 { font-size: 1rem; margin: 0 0 0.75rem 0; }",
        ".row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }",
        ".node { background: #e8f4fc; border: 1px solid #7eb8da; padding: 0.35rem 0.6rem; border-radius: 6px; font-size: 0.8rem; max-width: 100%; word-break: break-all; }",
        ".arrow { color: #666; font-weight: bold; }",
        ".tag { font-size: 0.75rem; color: #555; }",
        "</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class=\"tag\">seed={html.escape(str(seed))} &middot; {doc.get('num_chains', len(chains))} chains</p>",
        "<pre class=\"meta\">" + html.escape(json.dumps(constraints, indent=2)) + "</pre>",
    ]

    for ch in chains:
        idx = ch.get("index", 0)
        endpoints = ch.get("endpoints", [])
        pattern = ch.get("pattern", "")
        meta = ch.get("metadata", {})
        parts.append('<section class="chain">')
        parts.append(f"<h2>Chain {idx}</h2>")
        parts.append(f'<p class="tag">pattern={html.escape(str(pattern))} &middot; steps={len(endpoints)}</p>')
        if meta:
            parts.append(f'<p class="tag">{html.escape(json.dumps(meta, ensure_ascii=False))}</p>')
        parts.append('<div class="row">')
        for j, ep in enumerate(endpoints):
            if j:
                parts.append('<span class="arrow">→</span>')
            parts.append(f'<span class="node">{html.escape(ep)}</span>')
        parts.append("</div></section>")

    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
