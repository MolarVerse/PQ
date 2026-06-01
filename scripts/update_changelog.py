#!/usr/bin/env python3
"""Render changelog fragments and stamp the new release version.

Workflow:
  1. Collate every `changes/<slug>.<type>.md` fragment into the
     `## Next Release` section of `CHANGELOG.md`, grouped by section.
  2. Stamp the section with `## [<version>](...) - <date>` and move the
     `<!-- insertion marker -->` above it.
  3. Delete the consumed fragments so they don't reappear next release.

Usage: update_changelog.py <version>
"""
import os
import re
import sys
from datetime import date
from pathlib import Path

# Fragment type -> CHANGELOG section heading
SECTIONS = {
    "bugfix":      "Bug Fixes",
    "build":       "Build",
    "ci":          "CI",
    "internal":    "Internal",
    "test":        "Tests",
    "enhancement": "Enhancements",
    "doc":         "Documentation",
}

# Section order in the rendered changelog
ORDER = [
    "Enhancements",
    "Bug Fixes",
    "Build",
    "CI",
    "Internal",
    "Tests",
    "Documentation",
]

ROOT = Path(__file__).resolve().parent.parent
CHANGES_DIR = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"

FRAGMENT_RE = re.compile(r"^([^.]+)\.([^.]+)\.md$")


def read_fragments():
    """Group fragment bodies by section. Returns dict[section] -> [bullets...]."""
    if not CHANGES_DIR.is_dir():
        return {}, []
    grouped = {section: [] for section in ORDER}
    consumed = []
    for path in sorted(CHANGES_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        m = FRAGMENT_RE.match(path.name)
        if not m:
            sys.exit(f"unexpected fragment name: {path.name}")
        _, kind = m.groups()
        if kind not in SECTIONS:
            sys.exit(
                f"unknown fragment type '{kind}' in {path.name}; "
                f"allowed: {sorted(SECTIONS)}"
            )
        grouped[SECTIONS[kind]].append(path.read_text(encoding="utf-8").rstrip())
        consumed.append(path)
    return grouped, consumed


def render_fragments_into_next_release(lines, grouped):
    """Insert grouped fragments under '## Next Release', merging with any
    existing subsections of the same name."""
    # Find Next Release header and the next top-level header (or insertion marker)
    next_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if next_idx is None and re.match(r"^##\s+Next Release\s*$", line):
            next_idx = i
            continue
        if next_idx is not None and (
            re.match(r"^##\s+\[", line) or "<!-- insertion marker -->" in line
        ):
            end_idx = i
            break

    if next_idx is None:
        sys.exit("could not find '## Next Release' in CHANGELOG.md")

    # Parse existing subsections inside Next Release so we can append to them
    body = lines[next_idx + 1:end_idx]
    existing = {section: [] for section in ORDER}
    current = None
    for raw in body:
        m = re.match(r"^###\s+(.+?)\s*$", raw)
        if m:
            current = m.group(1).strip()
            if current not in existing:
                existing[current] = []
            continue
        if current and raw.strip():
            existing[current].append(raw)

    # Merge fragments
    for section, bullets in grouped.items():
        existing[section].extend(bullets)

    # Re-render
    rendered = []
    for section in ORDER:
        bullets = [b for b in existing[section] if b.strip()]
        if not bullets:
            continue
        rendered.append(f"### {section}")
        rendered.append("")
        rendered.extend(bullets)
        rendered.append("")

    return lines[:next_idx + 1] + [""] + rendered + lines[end_idx:]


def stamp_release(lines, version, repo):
    """Replace the '## Next Release' header with the versioned header and
    move the insertion marker above it."""
    new_header = (
        f"## [{version}](https://github.com/{repo}/releases/tag/{version}) "
        f"- {date.today().isoformat()}"
    )
    out = []
    stamped = False
    for line in lines:
        if not stamped and re.match(r"^##\s+Next Release\s*$", line):
            out.append("<!-- insertion marker -->")
            out.append(new_header)
            stamped = True
            continue
        if stamped and "<!-- insertion marker -->" in line:
            continue
        out.append(line)
    if not stamped:
        sys.exit("could not stamp release: '## Next Release' not found")
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <version>")
    if not CHANGELOG.exists():
        sys.exit("CHANGELOG.md not found")

    version = sys.argv[1]
    repo = os.getenv("GITHUB_REPOSITORY", "MolarVerse/PQ")

    grouped, consumed = read_fragments()
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    lines = render_fragments_into_next_release(lines, grouped)
    lines = stamp_release(lines, version, repo)

    CHANGELOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in consumed:
        path.unlink()

    print(
        f"CHANGELOG.md updated for {version}; "
        f"consumed {len(consumed)} fragment(s)"
    )


if __name__ == "__main__":
    main()
