#!/usr/bin/env python3
"""Require audience-qualified changelog fragments in a regular PR."""

import subprocess
import sys
from collections import Counter
from pathlib import Path

from changelog_fragments import (
    AUDIENCE_SECTIONS,
    FRAGMENT_RE,
    FragmentError,
    load_fragments,
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


def fragment_section_counts(fragments):
    """Count bullets by audience and rendered changelog section."""
    counts = Counter()
    for fragment in fragments:
        counts[(fragment.audience, fragment.section)] += len(fragment.entries)
    return counts


def fragment_counts_at_ref(ref):
    """Load fragment bullet counts directly from a Git tree."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", "changes"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-tree failed")

    counts = Counter()
    for relative_path in result.stdout.splitlines():
        audience = fragment_audience(relative_path)
        if audience is None:
            continue

        name = Path(relative_path).name
        section = parse_fragment_name(name, audience)
        show = subprocess.run(
            ["git", "show", f"{ref}:{relative_path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if show.returncode != 0:
            raise RuntimeError(show.stderr.strip() or "git show failed")
        entries = show.stdout.splitlines()
        if not entries or any(
            not entry.startswith("- ") or not entry[2:].strip()
            for entry in entries
        ):
            raise FragmentError(f"invalid fragment contents in {relative_path}")
        counts[(audience, section)] += len(entries)

    return counts


def validate_pr_changes(changes, root=ROOT, base_fragment_counts=None):
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

    deleted_fragments = [
        path for status, path in fragment_changes if status == "D"
    ]
    backlog_deletions = [
        path
        for path in deleted_fragments
        if path.endswith(".pre-fragment-backlog.md")
    ]
    forbidden_changes = [
        f"D {path}" for path in deleted_fragments if path not in backlog_deletions
    ]
    if backlog_deletions:
        if base_fragment_counts is None:
            forbidden_changes.extend(f"D {path}" for path in backlog_deletions)
        else:
            try:
                head_counts = fragment_section_counts(
                    load_fragments(root / "changes")
                )
            except (FragmentError, OSError) as error:
                errors.append(str(error))
                head_counts = Counter()

            missing = base_fragment_counts - head_counts
            if missing:
                missing_summary = ", ".join(
                    f"{audience}/{section}: {count}"
                    for (audience, section), count in sorted(missing.items())
                )
                errors.append(
                    "backlog cleanup must preserve every audience and section "
                    f"entry; missing {missing_summary}"
                )

    forbidden_changes.extend(
        f"{status} {path}"
        for status, path in fragment_changes
        if status not in {"A", "M", "D"}
    )
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
        changes = changed_files(base, head)
        base_counts = None
        if any(status == "D" for status, _ in changes):
            base_counts = fragment_counts_at_ref(base)
        errors = validate_pr_changes(changes, base_fragment_counts=base_counts)
    except (FragmentError, RuntimeError) as error:
        sys.exit(str(error))

    if errors:
        sys.exit("\n".join(f"- {error}" for error in errors))

    print("pull request contains valid changelog fragments")


if __name__ == "__main__":
    main()
