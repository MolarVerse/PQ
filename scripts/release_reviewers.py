#!/usr/bin/env python3
"""Extract human PR authors from generated GitHub release notes."""

import re
import sys


ENTRY_RE = re.compile(
    r"^[*-] .* by @(?P<login>[A-Za-z0-9-]+(?:\[bot\])?) in "
    r"https://github\.com/(?P<repository>[^/]+/[^/]+)/pull/[0-9]+$"
)


def extract_reviewers(notes, repository):
    """Return unique, non-bot PR authors in release-note order."""
    reviewers = []
    seen = set()

    for line in notes.splitlines():
        match = ENTRY_RE.match(line)
        if not match or match.group("repository") != repository:
            continue

        login = match.group("login")
        if login.endswith("[bot]") or login in seen:
            continue

        seen.add(login)
        reviewers.append(login)

    return reviewers


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <owner/repository>")

    notes = sys.stdin.read()
    for reviewer in extract_reviewers(notes, sys.argv[1]):
        print(reviewer)


if __name__ == "__main__":
    main()
