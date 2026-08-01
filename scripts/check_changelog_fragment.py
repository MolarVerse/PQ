#!/usr/bin/env python3
"""Require audience-qualified changelog fragments in a regular PR."""

import subprocess
import sys
from pathlib import Path

from changelog_fragments import (
    AUDIENCE_SECTIONS,
    FRAGMENT_RE,
    FragmentError,
    parse_fragment_name,
    read_fragment_entries,
)


ROOT = Path(__file__).resolve().parent.parent
PROTECTED_CHANGELOGS = {"CHANGELOG.md", "DEV-CHANGELOG.md"}


def changed_files(base, head):
    """Return status/path pairs for changes introduced after the base."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--no-renames",
            f"{base}...{head}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    changes = []
    for line in result.stdout.splitlines():
        status, path = line.split("\t", maxsplit=1)
        changes.append((status, path))
    return changes


def fragment_audience(relative_path):
    """Return the audience for a changes/<audience>/<file>.md path, if any."""
    parts = Path(relative_path).parts
    if len(parts) != 3 or parts[0] != "changes":
        return None
    if parts[1] not in AUDIENCE_SECTIONS:
        return None
    if not parts[2].endswith(".md") or parts[2] == "README.md":
        return None
    return parts[1]


def validate_pr_changes(changes, root=ROOT):
    """Return validation errors for a regular pull request."""
    errors = []
    direct_edits = sorted(
        path for _, path in changes if path in PROTECTED_CHANGELOGS
    )
    if direct_edits:
        errors.append(
            "regular pull requests must not edit "
            + " or ".join(direct_edits)
            + " directly"
        )

    fragment_changes = [
        (status, path)
        for status, path in changes
        if fragment_audience(path) is not None
    ]
    editable_fragments = [
        (status, path)
        for status, path in fragment_changes
        if status in {"A", "M"}
    ]

    if not fragment_changes:
        errors.append(
            "regular pull requests must add or update at least one "
            "changelog fragment under changes/user/ or changes/developer/"
        )

    forbidden_changes = [
        f"{status} {path}"
        for status, path in fragment_changes
        if status not in {"A", "M"}
    ]
    if forbidden_changes:
        errors.append(
            "regular pull requests must not delete or replace changelog "
            "fragments: "
            + ", ".join(forbidden_changes)
        )

    for _, relative_path in editable_fragments:
        audience = fragment_audience(relative_path)
        name = Path(relative_path).name
        if not FRAGMENT_RE.match(name):
            errors.append(
                f"invalid fragment name '{name}'; expected "
                "<category>.<slug>.md"
            )
            continue
        try:
            parse_fragment_name(name, audience)
            read_fragment_entries(root / relative_path)
        except (FragmentError, OSError) as error:
            errors.append(str(error))

    return errors


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: {sys.argv[0]} <base> [head]")

    base = sys.argv[1]
    head = sys.argv[2] if len(sys.argv) == 3 else "HEAD"

    try:
        errors = validate_pr_changes(changed_files(base, head))
    except RuntimeError as error:
        sys.exit(str(error))

    if errors:
        sys.exit("\n".join(f"- {error}" for error in errors))

    print("pull request contains valid changelog fragments")


if __name__ == "__main__":
    main()
