#!/usr/bin/env python3
"""Require audience-qualified changelog fragments in a regular PR."""

import subprocess
import sys
from pathlib import Path

from changelog_fragments import (
    FRAGMENT_RE,
    FragmentError,
    parse_fragment_name,
    read_fragment_entry,
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
        if path.startswith("changes/")
        and path.endswith(".md")
        and path != "changes/README.md"
    ]
    added_fragments = [
        path for status, path in fragment_changes if status == "A"
    ]

    if not added_fragments:
        errors.append(
            "regular pull requests must add at least one changelog fragment"
        )

    non_added = [
        f"{status} {path}"
        for status, path in fragment_changes
        if status != "A"
    ]
    if non_added:
        errors.append(
            "existing changelog fragments are immutable: "
            + ", ".join(non_added)
        )

    for relative_path in added_fragments:
        name = Path(relative_path).name
        if not FRAGMENT_RE.match(name):
            errors.append(
                f"invalid fragment name '{name}'; expected "
                "<slug>.<user|developer>.<category>.md"
            )
        else:
            try:
                parse_fragment_name(name)
                read_fragment_entry(root / relative_path)
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
