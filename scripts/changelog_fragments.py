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

LEGACY_SECTIONS = {
    "bugfix": "Bug Fixes",
    "build": "Build",
    "ci": "CI",
    "internal": "Internal",
    "test": "Tests",
    "enhancement": "Enhancements",
    "doc": "Documentation",
}

FRAGMENT_RE = re.compile(
    r"^(?P<slug>[a-z0-9][a-z0-9-]*)\."
    r"(?P<audience>user|developer)\."
    r"(?P<category>[a-z]+)\.md$"
)
LEGACY_FRAGMENT_RE = re.compile(
    r"^(?P<slug>[^.]+)\.(?P<category>[^.]+)\.md$"
)
MAX_ENTRY_LENGTH = 240


class FragmentError(ValueError):
    """Raised when a changelog fragment is malformed."""


@dataclass(frozen=True)
class Fragment:
    path: Path
    audience: str
    section: str
    entry: str


def parse_fragment_name(name):
    """Return audience and section encoded by a fragment filename."""
    match = FRAGMENT_RE.match(name)
    if not match:
        raise FragmentError(
            f"invalid fragment name '{name}'; expected "
            "<slug>.<user|developer>.<category>.md"
        )

    audience = match.group("audience")
    category = match.group("category")
    sections = AUDIENCE_SECTIONS[audience]
    if category not in sections:
        allowed = ", ".join(sorted(sections))
        raise FragmentError(
            f"invalid {audience} category '{category}' in {name}; "
            f"allowed: {allowed}"
        )

    return audience, sections[category]


def read_fragment_entry(path):
    """Read one concise Markdown bullet from a new-style fragment."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if len(lines) != 1 or not lines[0].startswith("- ") or not lines[0][2:].strip():
        raise FragmentError(
            f"{path.name} must contain exactly one non-empty Markdown bullet"
        )
    if len(lines[0]) > MAX_ENTRY_LENGTH:
        raise FragmentError(
            f"{path.name} exceeds the {MAX_ENTRY_LENGTH}-character limit"
        )
    if not text.endswith("\n"):
        raise FragmentError(f"{path.name} must end with a newline")

    return lines[0]


def load_fragments(changes_dir):
    """Load new audience fragments and compatible legacy fragments."""
    if not changes_dir.is_dir():
        return []

    fragments = []
    for path in sorted(changes_dir.glob("*.md")):
        if path.name == "README.md":
            continue

        if FRAGMENT_RE.match(path.name):
            audience, section = parse_fragment_name(path.name)
            entry = read_fragment_entry(path)
        else:
            match = LEGACY_FRAGMENT_RE.match(path.name)
            if not match:
                raise FragmentError(f"unexpected fragment name: {path.name}")

            category = match.group("category")
            if category not in LEGACY_SECTIONS:
                allowed = ", ".join(sorted(LEGACY_SECTIONS))
                raise FragmentError(
                    f"unknown legacy category '{category}' in {path.name}; "
                    f"allowed: {allowed}"
                )

            audience = "developer"
            section = LEGACY_SECTIONS[category]
            entry = path.read_text(encoding="utf-8").rstrip()
            if not entry:
                raise FragmentError(f"{path.name} must not be empty")

        fragments.append(Fragment(path, audience, section, entry))

    return fragments
