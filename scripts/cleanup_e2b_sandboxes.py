#!/usr/bin/env python3
"""
Cleanup helper: kill all running E2B sandboxes for the current account.

Notes
- Loads credentials from .env (E2B_API_KEY). If not present, falls back to env.
- Uses the paginator correctly: while has_next, call next_items() and kill each.
- Intended to be called at the end of smoke runs to avoid hitting concurrency limits.
"""

from __future__ import annotations

from dotenv import load_dotenv

from e2b import Sandbox


def main() -> None:
    # Load API key from local .env if present
    load_dotenv(dotenv_path=".env")

    total = 0
    paginator = Sandbox.list(limit=100)
    while paginator.has_next:
        items = paginator.next_items()
        if not items:
            break
        for info in items:
            sid = info.sandbox_id
            try:
                sbx = Sandbox.connect(sandbox_id=sid)
                sbx.kill()
                total += 1
                print(f"killed {sid}")
            except Exception as exc:  # best effort
                print(f"kill failed for {sid}: {exc}")
    print(f"Total killed: {total}")


if __name__ == "__main__":
    main()

