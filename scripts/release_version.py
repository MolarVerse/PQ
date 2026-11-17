#!/usr/bin/env python3
"""Resolve the release version from the release branch or the hotfix PR title."""

import argparse
import re
import subprocess
import sys


TAG = re.compile(r"\S+\s+refs/tags/v([0-9]+)\.([0-9]+)\.([0-9]+)")
HOTFIX_TITLE = re.compile(r"Hotfix-([0-9]+\.[0-9]+\.[0-9]+)(?=$|[\s:])")


def latest_tag():
    remote = subprocess.check_output(
        ["git", "ls-remote", "--tags", "--refs", "origin"], text=True
    )
    tags = [
        tuple(map(int, match.groups()))
        for line in remote.splitlines()
        if (match := TAG.fullmatch(line))
    ]
    if not tags:
        raise ValueError("could not infer the latest release tag")
    return max(tags)


def resolve_version(head_ref, prepare, title):
    if re.fullmatch(r"release/v[0-9]+\.[0-9]+\.[0-9]+", head_ref):
        version = head_ref.removeprefix("release/")
        if title is not None and title != f"Release-{version[1:]}":
            raise ValueError(f"Release PR title must be Release-{version[1:]}.")
    elif head_ref.startswith("hotfix/") and head_ref != "hotfix/":
        match = HOTFIX_TITLE.match(title or "")
        if not match:
            raise ValueError(
                "Hotfix PR title must start with Hotfix-X.Y.Z, "
                "e.g. 'Hotfix-1.0.1: describe the fix'."
            )
        version = f"v{match.group(1)}"
        if prepare:
            major, minor, patch = latest_tag()
            expected = f"v{major}.{minor}.{patch + 1}"
            if version != expected:
                raise ValueError(
                    f"Hotfix version must be {expected} (latest tag + 1), "
                    f"not {version}."
                )
    else:
        raise ValueError("release branches must use release/vX.Y.Z or hotfix/<name>")
    if prepare:
        remote = subprocess.check_output(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{version}"], text=True
        )
        if remote.strip():
            raise ValueError(f"{version} already exists; choose a new release version")
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
