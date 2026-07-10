#!/usr/bin/env python3
"""Stamp a release version into CHANGELOG.md, auto-generating the section
from conventional commits when no curated content exists yet.

Behavior:

  1. If `## Next Release` already contains subsections with bullets, that
     content is left as-is and stamped with the new version. Existing
     legacy `changes/<slug>.<type>.md` fragments (if any) are folded in
     under the matching section so they aren't lost.
  2. If `## Next Release` is empty (no subsections), git-cliff is invoked
     to render fresh bullets from conventional commits since the previous
     tag, using cliff.toml at the repo root.
  3. The section header is then rewritten to
     `## [<version>](...) - <date>` and the `<!-- insertion marker -->`
     is moved above it. Consumed fragments are unlinked.

Usage: update_changelog.py <version>
"""
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# Fragment type -> CHANGELOG section heading. Legacy: kept so a fragment
# left over from the deprecated changes/ flow still ends up in the right
# section.
SECTIONS = {
    "bugfix":      "Bug Fixes",
    "build":       "Build",
    "ci":          "CI",
    "internal":    "Internal",
    "test":        "Tests",
    "enhancement": "Enhancements",
    "doc":         "Documentation",
}

# Section order in the rendered changelog (matches cliff.toml).
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
CHANGELOG = ROOT / "CHANGELOG.md"
CLIFF_TOML = ROOT / "cliff.toml"

FRAGMENT_RE = re.compile(r"^([^.]+)\.([^.]+)\.md$")


def read_fragments():
    """Group fragment bodies by section (legacy `changes/` flow)."""
    if not CHANGES_DIR.is_dir():
        return {s: [] for s in ORDER}, []
    grouped = {s: [] for s in ORDER}
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


def split_next_release(lines):
    """Locate the `## Next Release` block and split the file around it.

    Returns (head, body, tail) where head/tail are the lines before/after
    the block (inclusive of `## Next Release` and `<!-- insertion marker -->`
    respectively), and body is the bullet content between them.
    """
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
    return lines[:next_idx + 1], lines[next_idx + 1:end_idx], lines[end_idx:]


def parse_subsections(body_lines):
    """Parse subsections inside the `## Next Release` block.

    Returns dict[section_heading] -> list of raw bullet lines.
    """
    sections = {s: [] for s in ORDER}
    current = None
    for raw in body_lines:
        m = re.match(r"^###\s+(.+?)\s*$", raw)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current and raw.strip():
            sections[current].append(raw)
    return sections


def render_sections(sections):
    """Render the per-section dict back into markdown lines."""
    out = []
    for section in ORDER:
        bullets = [b for b in sections.get(section, []) if b.strip()]
        if not bullets:
            continue
        out.append(f"### {section}")
        out.append("")
        out.extend(bullets)
        out.append("")
    # Any custom section not in ORDER goes at the end, preserving order
    # of first appearance.
    for section, bullets in sections.items():
        if section in ORDER:
            continue
        bullets = [b for b in bullets if b.strip()]
        if not bullets:
            continue
        out.append(f"### {section}")
        out.append("")
        out.extend(bullets)
        out.append("")
    return out


def run_git_cliff():
    """Generate per-section bullets via git-cliff for commits since the
    last tag. Returns dict[section_heading] -> list of raw bullet lines."""
    if not CLIFF_TOML.is_file():
        sys.exit(f"cliff.toml not found at {CLIFF_TOML}")
    result = subprocess.run(
        [
            "git-cliff",
            "--unreleased",
            "--strip", "all",
            "--config", str(CLIFF_TOML),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        sys.exit(
            "git-cliff failed (exit "
            f"{result.returncode}): {result.stderr.strip()}"
        )
    return parse_subsections(result.stdout.splitlines())


def stamp_release(head_lines, body_lines, tail_lines, version, repo):
    """Replace the `## Next Release` heading with the versioned heading
    and put the `<!-- insertion marker -->` above it. Strips any prior
    insertion-marker line out of the tail."""
    new_header = (
        f"## [{version}](https://github.com/{repo}/releases/tag/{version}) "
        f"- {date.today().isoformat()}"
    )
    # Replace the last line of head ("## Next Release") with the marker +
    # the versioned header.
    head = head_lines[:-1] + ["<!-- insertion marker -->", new_header, ""]
    # Drop any lingering insertion-marker lines in the tail.
    tail = [line for line in tail_lines if "<!-- insertion marker -->" not in line]
    return head + body_lines + tail


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <version>")
    if not CHANGELOG.exists():
        sys.exit("CHANGELOG.md not found")

    version = sys.argv[1]
    repo = os.getenv("GITHUB_REPOSITORY", "MolarVerse/PQ")

    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    head, body, tail = split_next_release(lines)

    curated = parse_subsections(body)
    has_curated_content = any(
        any(line.strip() for line in bullets)
        for bullets in curated.values()
    )

    fragment_bullets, consumed_fragments = read_fragments()

    if has_curated_content:
        # Preserve curated content; only merge in any leftover fragments.
        sections = curated
        for s, bullets in fragment_bullets.items():
            sections.setdefault(s, []).extend(bullets)
        source = "curated CHANGELOG.md Next Release content"
        if any(fragment_bullets.values()):
            source += " + legacy fragments"
    else:
        # No curated content: auto-generate from conventional commits.
        sections = run_git_cliff()
        for s, bullets in fragment_bullets.items():
            sections.setdefault(s, []).extend(bullets)
        source = "git-cliff (commits since the last tag)"
        if any(fragment_bullets.values()):
            source += " + legacy fragments"

    body_out = [""] + render_sections(sections)
    out_lines = stamp_release(head, body_out, tail, version, repo)

    CHANGELOG.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    for path in consumed_fragments:
        path.unlink()

    print(
        f"CHANGELOG.md stamped as {version} from {source}; "
        f"consumed {len(consumed_fragments)} fragment(s)"
    )


if __name__ == "__main__":
    main()
