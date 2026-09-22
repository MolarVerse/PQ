#!/usr/bin/env python3
"""Parse a @pq-bot mention into a task file plus an allowlisted model.

Usage: pq_bot_parse.py <issue-or-pr-number> <comment-body> <author> <outdir>

Writes <outdir>/pq-task.txt (task text, may be multi-line) and prints a
single `MODEL=<provider/model>` line (empty value = account default).
Only models on the allowlist below are ever honored; anything else
falls back to the default. Model IDs come from the environment so no
plan-specific names are hardcoded here.
"""

import os
import re
import sys

ALLOWLIST = {
    "cheap": os.environ.get("PQ_BOT_MODEL_CHEAP", ""),
    "fast": os.environ.get("PQ_BOT_MODEL_FAST", ""),
    "smart": os.environ.get("PQ_BOT_MODEL_SMART", ""),
    "review": os.environ.get("PQ_BOT_MODEL_REVIEW", ""),
}

COMMANDS = (
    "test",
    "fix",
    "review",
    "cleanup",
    "format",
    "rebase",
    "rerun",
    "triage",
    "repro",
    "docs",
    "deps",
    "perf",
)


def main() -> int:
    number, body, author, outdir = sys.argv[1:5]

    match = re.search(r"@pq-bot\s+(\w+)(.*)", body, re.DOTALL)
    command = (match.group(1) if match else "").lower()
    rest = (match.group(2) if match else "").strip().splitlines()[0:1]
    detail = rest[0].strip() if rest else ""

    if command not in COMMANDS:
        command = "triage"

    model = ""
    with_match = re.search(r"\bwith\s+(\w+)\b", detail, re.IGNORECASE)
    if with_match:
        model = ALLOWLIST.get(with_match.group(1).lower(), "")
    model = re.sub(r"[^A-Za-z0-9/.:_-]", "", model)

    task = (
        f"Repository task from @{author} (thread #{number}).\n"
        f"Command: {command}\n"
        f"Detail: {detail}\n"
        "Follow your system prompt. If the request is outside your "
        "tiers, refuse with one sentence."
    )
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "pq-task.txt"), "w") as handle:
        handle.write(task)

    print(f"MODEL={model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
