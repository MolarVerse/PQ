#!/usr/bin/env python3
"""Check release tag and branch availability without treating Git errors as absence."""

import argparse
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.version):
        parser.error("version must use vX.Y.Z")
    try:
        references = subprocess.check_output(
            [
                "git",
                "ls-remote",
                "--refs",
                "origin",
                f"refs/tags/{args.version}",
                f"refs/heads/release/{args.version}",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"Cannot check release refs: {error}")
    if references.strip():
        sys.exit(f"Release tag or branch for {args.version} already exists")
    print(f"Release tag and branch for {args.version} are available.")


if __name__ == "__main__":
    main()
