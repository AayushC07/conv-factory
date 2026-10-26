"""Verify OPENAI_API_KEY with a minimal chat completion.

Usage (from repo root):
  OPENAI_API_KEY=sk-... python scripts/test_openai_key.py
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> int:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        print("OPENAI_API_KEY is missing or empty. Set it in the environment.", file=sys.stderr)
        return 1

    model = os.environ.get("OPENAI_TEST_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with exactly the word "ok".'}],
            max_tokens=16,
        )
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 2

    text = (response.choices[0].message.content or "").strip()
    print(f"OK - API key works (model={model}).")
    print(f"Sample reply: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
