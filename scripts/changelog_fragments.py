"""Parse changelog fragments shared by PR checks and release tooling."""

from dataclasses import dataclass
from pathlib import Path
import re


USER_SECTIONS = {
    "enhancement": "New Features",
    "change": "Changes",
    "bugfix": "Bug Fixes",
    "performance": "Performance",
    "compatibility": "Build and Compatibility",
    "documentation": "Documentation",
}

DEVELOPER_SECTIONS = {
    "enhancement": "Enhancements",
    "bugfix": "Bug Fixes",
    "performance": "Performance",
    "build": "Build",
    "ci": "CI",
    "test": "Tests",
    "internal": "Internal",
    "documentation": "Documentation",
}

AUDIENCE_SECTIONS = {
    "user": USER_SECTIONS,
    "developer": DEVELOPER_SECTIONS,
}

USER_ORDER = list(USER_SECTIONS.values())
DEVELOPER_ORDER = [
    "Breaking Changes",
    *DEVELOPER_SECTIONS.values(),
]

# <category>.<slug>.md, living under changes/<user|developer>/. The audience
# is carried by the parent directory, not the filename.
FRAGMENT_RE = re.compile(
    r"^(?P<category>[a-z]+)\.(?P<slug>[a-z0-9][a-z0-9-]*)\.md$"
)
MAX_ENTRY_LENGTH = 240


class FragmentError(ValueError):
    """Raised when a changelog fragment is malformed."""


@dataclass(frozen=True)
class Fragment:
    path: Path
    audience: str
    section: str
    entries: list


def parse_fragment_name(name, audience):
    """Return the changelog section encoded by a fragment filename."""
    match = FRAGMENT_RE.match(name)
    if not match:
        raise FragmentError(
            f"invalid fragment name '{name}'; expected <category>.<slug>.md"
        )

    category = match.group("category")
    sections = AUDIENCE_SECTIONS[audience]
    if category not in sections:
        allowed = ", ".join(sorted(sections))
        raise FragmentError(
            f"invalid {audience} category '{category}' in {name}; "
            f"allowed: {allowed}"
        )

    return sections[category]


def read_fragment_entries(path):
    """Read one or more concise Markdown bullets from a fragment."""
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise FragmentError(f"{path.name} must end with a newline")

    lines = text.rstrip("\n").split("\n")
    entries = []
    for line in lines:
        if not line.startswith("- ") or not line[2:].strip():
            raise FragmentError(
                f"{path.name} must contain only Markdown bullets ('- ...'), "
                "with no blank lines"
            )
        if len(line) > MAX_ENTRY_LENGTH:
            raise FragmentError(
                f"{path.name} has a bullet exceeding the "
                f"{MAX_ENTRY_LENGTH}-character limit"
            )
        entries.append(line)

    if not entries:
        raise FragmentError(f"{path.name} must contain at least one bullet")

    return entries


def load_fragments(changes_dir):
    """Load fragments from changes/user/ and changes/developer/."""
    fragments = []
    for audience in AUDIENCE_SECTIONS:
        audience_dir = changes_dir / audience
        if not audience_dir.is_dir():
            continue

        for path in sorted(audience_dir.glob("*.md")):
            if path.name == "README.md":
                continue

            section = parse_fragment_name(path.name, audience)
            entries = read_fragment_entries(path)
            fragments.append(Fragment(path, audience, section, entries))

    return fragments
