#!/usr/bin/env python3
"""Stamp user and developer changelogs for a release.

`CHANGELOG.md` is curated in the release pull request and contains only
user-visible changes. `DEV-CHANGELOG.md` is generated from conventional
commits at release time and contains the complete technical record.

Regular pull requests do not edit either file. This keeps changelog curation
out of the normal merge path while still requiring reviewed user release notes.

Usage:
    update_changelog.py --check
    update_changelog.py <version>
"""

import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# Fragment type -> developer changelog section. Fragments are deprecated, but
# this keeps fragments from older branches from being lost when they merge.
SECTIONS = {
    "bugfix": "Bug Fixes",
    "build": "Build",
    "ci": "CI",
    "internal": "Internal",
    "test": "Tests",
    "enhancement": "Enhancements",
    "doc": "Documentation",
}

ORDER = [
    "Breaking Changes",
    "Enhancements",
    "Bug Fixes",
    "Performance",
    "Build",
    "CI",
    "Tests",
    "Internal",
    "Documentation",
]

ROOT = Path(__file__).resolve().parent.parent
CHANGES_DIR = ROOT / "changes"
USER_CHANGELOG = ROOT / "CHANGELOG.md"
DEV_CHANGELOG = ROOT / "DEV-CHANGELOG.md"
CLIFF_TOML = ROOT / "cliff.toml"

FRAGMENT_RE = re.compile(r"^([^.]+)\.([^.]+)\.md$")
USER_BULLET_RE = re.compile(r"^\s*-\s+\S")


def read_fragments():
    """Group legacy fragment bodies by developer changelog section."""
    if not CHANGES_DIR.is_dir():
        return {section: [] for section in ORDER}, []

    grouped = {section: [] for section in ORDER}
    consumed = []
    for path in sorted(CHANGES_DIR.glob("*.md")):
        if path.name == "README.md":
            continue

        match = FRAGMENT_RE.match(path.name)
        if not match:
            sys.exit(f"unexpected fragment name: {path.name}")

        _, kind = match.groups()
        if kind not in SECTIONS:
            sys.exit(
                f"unknown fragment type '{kind}' in {path.name}; "
                f"allowed: {sorted(SECTIONS)}"
            )

        grouped[SECTIONS[kind]].append(
            path.read_text(encoding="utf-8").rstrip()
        )
        consumed.append(path)

    return grouped, consumed


def split_next_release(lines, changelog):
    """Return the file head, Next Release body, and release-history tail."""
    next_index = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        if next_index is None and re.match(r"^##\s+Next Release\s*$", line):
            next_index = index
            continue
        if next_index is not None and (
            re.match(r"^##\s+\[", line)
            or "<!-- insertion marker -->" in line
        ):
            end_index = index
            break

    if next_index is None:
        sys.exit(f"could not find '## Next Release' in {changelog}")

    return (
        lines[: next_index + 1],
        lines[next_index + 1 : end_index],
        lines[end_index:],
    )


def parse_subsections(body_lines):
    """Parse `###` sections into a mapping of headings to bullet lines."""
    sections = {section: [] for section in ORDER}
    current = None

    for raw in body_lines:
        match = re.match(r"^###\s+(.+?)\s*$", raw)
        if match:
            current = match.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current and raw.strip():
            sections[current].append(raw)

    return sections


def render_sections(sections):
    """Render developer changelog sections in a stable order."""
    output = []

    for section in ORDER:
        bullets = [line for line in sections.get(section, []) if line.strip()]
        if not bullets:
            continue
        output.extend([f"### {section}", "", *bullets, ""])

    for section, bullets in sections.items():
        if section in ORDER:
            continue
        bullets = [line for line in bullets if line.strip()]
        if not bullets:
            continue
        output.extend([f"### {section}", "", *bullets, ""])

    return output


def run_git_cliff():
    """Generate developer sections from commits since the latest tag."""
    if not CLIFF_TOML.is_file():
        sys.exit(f"cliff.toml not found at {CLIFF_TOML}")

    result = subprocess.run(
        [
            "git-cliff",
            "--unreleased",
            "--strip",
            "all",
            "--config",
            str(CLIFF_TOML),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(
            "git-cliff failed (exit "
            f"{result.returncode}): {result.stderr.strip()}"
        )

    return parse_subsections(result.stdout.splitlines())


def has_user_release_notes(body_lines):
    """Return whether the user changelog contains at least one bullet."""
    return any(USER_BULLET_RE.match(line) for line in body_lines)


def trim_blank_lines(lines):
    """Remove blank lines at both edges without changing inner formatting."""
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    return lines[start:end]


def stamp_release(head, body, tail, version, repo):
    """Move a Next Release body under a version while keeping Next Release."""
    release_header = (
        f"## [{version}](https://github.com/{repo}/releases/tag/{version}) "
        f"- {date.today().isoformat()}"
    )
    clean_body = trim_blank_lines(body)
    clean_tail = [
        line for line in tail if "<!-- insertion marker -->" not in line
    ]
    clean_tail = trim_blank_lines(clean_tail)

    output = [
        *trim_blank_lines(head),
        "",
        "<!-- insertion marker -->",
        release_header,
        "",
        *clean_body,
    ]
    if clean_tail:
        output.extend(["", *clean_tail])

    return output


def load_changelog(path):
    if not path.is_file():
        sys.exit(f"{path.name} not found")
    return path.read_text(encoding="utf-8").splitlines()


def check_user_changelog():
    lines = load_changelog(USER_CHANGELOG)
    _, body, _ = split_next_release(lines, USER_CHANGELOG.name)
    if not has_user_release_notes(body):
        sys.exit(
            "CHANGELOG.md needs at least one user-facing bullet under "
            "'## Next Release' before release"
        )
    print("CHANGELOG.md contains curated user release notes")


def update_changelogs(version):
    repo = os.getenv("GITHUB_REPOSITORY", "MolarVerse/PQ")

    user_lines = load_changelog(USER_CHANGELOG)
    user_head, user_body, user_tail = split_next_release(
        user_lines, USER_CHANGELOG.name
    )
    if not has_user_release_notes(user_body):
        sys.exit(
            "CHANGELOG.md needs curated user release notes before release"
        )

    dev_lines = load_changelog(DEV_CHANGELOG)
    dev_head, _, dev_tail = split_next_release(
        dev_lines, DEV_CHANGELOG.name
    )

    dev_sections = run_git_cliff()
    fragment_sections, consumed_fragments = read_fragments()
    for section, bullets in fragment_sections.items():
        dev_sections.setdefault(section, []).extend(bullets)

    user_output = stamp_release(
        user_head, user_body, user_tail, version, repo
    )
    dev_output = stamp_release(
        dev_head,
        render_sections(dev_sections),
        dev_tail,
        version,
        repo,
    )

    USER_CHANGELOG.write_text(
        "\n".join(user_output) + "\n", encoding="utf-8"
    )
    DEV_CHANGELOG.write_text(
        "\n".join(dev_output) + "\n", encoding="utf-8"
    )
    for path in consumed_fragments:
        path.unlink()

    print(
        f"stamped {USER_CHANGELOG.name} from curated release notes and "
        f"{DEV_CHANGELOG.name} from conventional commits; consumed "
        f"{len(consumed_fragments)} legacy fragment(s)"
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} --check | <version>")

    argument = sys.argv[1]
    if argument == "--check":
        check_user_changelog()
        return

    update_changelogs(argument)


if __name__ == "__main__":
    main()
