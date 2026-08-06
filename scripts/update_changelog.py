#!/usr/bin/env python3
"""Build and stamp user and developer changelogs for a release.

Regular pull requests add one audience-qualified fragment instead of editing
the shared changelog files. Release processing routes each fragment into the
matching changelog and preserves existing unreleased and released entries.

Usage:
    update_changelog.py --check
    update_changelog.py --check-prepared <version>
    update_changelog.py <version>
"""

import os
import re
import sys
from datetime import date
from pathlib import Path

from changelog_fragments import (
    DEVELOPER_ORDER,
    USER_ORDER,
    FragmentError,
    load_fragments,
)

ROOT = Path(__file__).resolve().parent.parent
CHANGES_DIR = ROOT / "changes"
USER_CHANGELOG = ROOT / "CHANGELOG.md"
DEV_CHANGELOG = ROOT / "DEV-CHANGELOG.md"

BULLET_RE = re.compile(r"^\s*-\s+\S")


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


def parse_subsections(body_lines, order):
    """Parse `###` sections into a mapping of headings to bullet lines."""
    sections = {section: [] for section in order}
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


def render_sections(sections, order):
    """Render changelog sections in a stable order."""
    output = []

    for section in order:
        bullets = [line for line in sections.get(section, []) if line.strip()]
        if not bullets:
            continue
        output.extend([f"### {section}", "", *bullets, ""])

    for section, bullets in sections.items():
        if section in order:
            continue
        bullets = [line for line in bullets if line.strip()]
        if not bullets:
            continue
        output.extend([f"### {section}", "", *bullets, ""])

    return output


def read_all_fragments():
    try:
        return load_fragments(CHANGES_DIR)
    except FragmentError as error:
        sys.exit(str(error))


def append_fragments(sections, fragments, audience):
    """Append fragments for one audience to parsed changelog sections."""
    for fragment in fragments:
        if fragment.audience == audience:
            sections.setdefault(fragment.section, []).extend(fragment.entries)


def has_release_notes(body_lines):
    """Return whether a changelog body contains at least one bullet."""
    return any(BULLET_RE.match(line) for line in body_lines)


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


def check_release_changelogs():
    fragments = read_all_fragments()

    user_lines = load_changelog(USER_CHANGELOG)
    _, user_body, _ = split_next_release(
        user_lines, USER_CHANGELOG.name
    )
    user_sections = parse_subsections(user_body, USER_ORDER)
    append_fragments(user_sections, fragments, "user")

    dev_lines = load_changelog(DEV_CHANGELOG)
    _, dev_body, _ = split_next_release(dev_lines, DEV_CHANGELOG.name)
    dev_sections = parse_subsections(dev_body, DEVELOPER_ORDER)
    append_fragments(dev_sections, fragments, "developer")

    user_notes = render_sections(user_sections, USER_ORDER)
    dev_notes = render_sections(dev_sections, DEVELOPER_ORDER)
    if not has_release_notes(user_notes) and not has_release_notes(dev_notes):
        sys.exit("the release needs at least one changelog entry")

    print("the release contains changelog entries")


def check_prepared_changelogs(version):
    """Check that a release is stamped and no fragments remain."""
    header_prefix = f"## [{version}]("
    user_lines = load_changelog(USER_CHANGELOG)
    dev_lines = load_changelog(DEV_CHANGELOG)

    if not any(line.startswith(header_prefix) for line in user_lines) and not any(
        line.startswith(header_prefix) for line in dev_lines
    ):
        sys.exit(
            f"release changelogs are not prepared for {version}; "
            f"run scripts/update_changelog.py {version}"
        )

    remaining = sorted(
        path
        for path in CHANGES_DIR.rglob("*.md")
        if path.name != "README.md"
    )
    if remaining:
        names = ", ".join(
            str(path.relative_to(CHANGES_DIR)) for path in remaining
        )
        sys.exit(f"release contains unprocessed changelog fragments: {names}")

    print(f"release changelogs are prepared for {version}")


def update_changelogs(version):
    repo = os.getenv("GITHUB_REPOSITORY", "MolarVerse/PQ")

    user_lines = load_changelog(USER_CHANGELOG)
    user_head, user_body, user_tail = split_next_release(
        user_lines, USER_CHANGELOG.name
    )

    dev_lines = load_changelog(DEV_CHANGELOG)
    dev_head, dev_body, dev_tail = split_next_release(
        dev_lines, DEV_CHANGELOG.name
    )

    fragments = read_all_fragments()
    user_sections = parse_subsections(user_body, USER_ORDER)
    dev_sections = parse_subsections(dev_body, DEVELOPER_ORDER)
    append_fragments(user_sections, fragments, "user")
    append_fragments(dev_sections, fragments, "developer")

    rendered_user_sections = render_sections(user_sections, USER_ORDER)
    rendered_dev_sections = render_sections(dev_sections, DEVELOPER_ORDER)
    has_user_notes = has_release_notes(rendered_user_sections)
    has_dev_notes = has_release_notes(rendered_dev_sections)
    if not has_user_notes and not has_dev_notes:
        sys.exit("the release needs at least one changelog entry")

    stamped = []
    if has_user_notes:
        user_output = stamp_release(
            user_head, rendered_user_sections, user_tail, version, repo
        )
        USER_CHANGELOG.write_text(
            "\n".join(user_output) + "\n", encoding="utf-8"
        )
        stamped.append(USER_CHANGELOG.name)

    if has_dev_notes:
        dev_output = stamp_release(
            dev_head,
            rendered_dev_sections,
            dev_tail,
            version,
            repo,
        )
        DEV_CHANGELOG.write_text(
            "\n".join(dev_output) + "\n", encoding="utf-8"
        )
        stamped.append(DEV_CHANGELOG.name)

    for fragment in fragments:
        fragment.path.unlink()

    print(
        f"stamped {' and '.join(stamped)}; "
        f"consumed {len(fragments)} changelog fragment(s)"
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check-prepared":
        check_prepared_changelogs(sys.argv[2])
        return

    if len(sys.argv) != 2:
        sys.exit(
            f"usage: {sys.argv[0]} --check | "
            "--check-prepared <version> | <version>"
        )

    argument = sys.argv[1]
    if argument == "--check":
        check_release_changelogs()
        return

    update_changelogs(argument)


if __name__ == "__main__":
    main()
