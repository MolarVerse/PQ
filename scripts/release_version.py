#!/usr/bin/env python3
"""Resolve release versions and persist the chosen hotfix version on its branch."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


STATE = Path(".github/release.json")
VERSION = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")


def resolve_version(head_ref, prepare, title):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not isinstance(state, dict):
        raise ValueError("invalid release metadata")
    if re.fullmatch(r"release/v[0-9]+\.[0-9]+\.[0-9]+", head_ref):
        version = head_ref.removeprefix("release/")
        if title is not None and title != f"Release-{version[1:]}":
            raise ValueError(f"Release PR title must be Release-{version[1:]}.")
    elif head_ref.startswith("hotfix/") and head_ref != "hotfix/":
        if state.get("branch") == head_ref:
            version = state.get("version", "")
        elif prepare:
            remote = subprocess.check_output(
                ["git", "ls-remote", "--tags", "--refs", "origin"], text=True
            )
            tags = [
                tuple(map(int, match.groups()))
                for line in remote.splitlines()
                if (
                    match := re.fullmatch(
                        r"\S+\s+refs/tags/v([0-9]+)\.([0-9]+)\.([0-9]+)", line
                    )
                )
            ]
            if not tags:
                raise ValueError("could not infer the latest release tag")
            major, minor, patch = max(tags)
            version = f"v{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(
                "hotfix version is not pinned; prepare its release branch first"
            )
    else:
        raise ValueError("release branches must use release/vX.Y.Z or hotfix/<name>")
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise ValueError("invalid pinned release version")
    if prepare:
        remote = subprocess.check_output(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{version}"], text=True
        )
        if remote.strip():
            raise ValueError(f"{version} already exists; choose a new release version")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(
            json.dumps({"branch": head_ref, "version": version}, indent=2) + "\n"
        )
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--title")
    args = parser.parse_args()
    try:
        print(resolve_version(args.head_ref, args.prepare, args.title))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
