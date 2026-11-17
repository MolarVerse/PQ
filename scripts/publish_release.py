#!/usr/bin/env python3
"""Create a release on HEAD, or verify a completed publication on retry."""

import argparse
import json
import os
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def git(*arguments):
    return subprocess.check_output(["git", *arguments], text=True).strip()


def remote_target(version):
    references = {}
    for line in git(
        "ls-remote",
        "--tags",
        "origin",
        f"refs/tags/{version}",
        f"refs/tags/{version}^{{}}",
    ).splitlines():
        sha, reference = line.split()
        references[reference] = sha
    return references.get(
        f"refs/tags/{version}^{{}}", references.get(f"refs/tags/{version}")
    )


def release_request(repository, version, data=None):
    base = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    path = f"/repos/{repository}/releases"
    if data is None:
        path += f"/tags/{version}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        base + path,
        headers=headers,
        data=None if data is None else json.dumps(data).encode(),
    )
    try:
        with urlopen(request, timeout=30) as response:
            release = json.load(response)
        if not isinstance(release, dict):
            raise ValueError("GitHub returned invalid release data")
        return release
    except HTTPError as error:
        if data is None and error.code == 404:
            return None
        raise


def verify_release(release, version):
    if (
        not isinstance(release, dict)
        or release.get("tag_name") != version
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or not release.get("published_at")
    ):
        raise ValueError(
            f"{version} is not a published stable release; inspect it before retrying"
        )


def publish(repository, version):
    target = git("rev-parse", "HEAD")
    existing_target = remote_target(version)
    if existing_target is not None and existing_target != target:
        raise ValueError(f"{version} already points to another commit")
    release = release_request(repository, version)
    if release is not None:
        verify_release(release, version)
        if existing_target is None:
            raise ValueError(
                f"existing release {version} has no tag; restore its original tag before retrying"
            )

    if existing_target is None:
        local_tag = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{version}"]
        )
        if local_tag.returncode == 1:
            git("tag", version, target)
        elif local_tag.returncode != 0:
            local_tag.check_returncode()
        elif git("rev-parse", f"refs/tags/{version}^{{commit}}") != target:
            raise ValueError(f"local tag {version} points to another commit")
        git("push", "origin", f"refs/tags/{version}")

    if remote_target(version) != target:
        raise ValueError(f"could not verify {version} on the release commit")
    if release is None:
        release = release_request(
            repository,
            version,
            {
                "tag_name": version,
                "target_commitish": target,
                "name": version,
                "draft": False,
                "prerelease": False,
                "generate_release_notes": True,
            },
        )
        verify_release(release, version)
        verify_release(release_request(repository, version), version)
    if remote_target(version) != target:
        raise ValueError(
            f"{version} moved away from the release commit during publication"
        )
    print(f"Verified published release {version} at {target}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("version")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("repository must use owner/name")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", args.version):
        parser.error("version must use vX.Y.Z")
    try:
        publish(args.repository, args.version)
    except HTTPError as error:
        sys.exit(f"Cannot publish release: GitHub returned HTTP {error.code}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Cannot publish release: {error}")


if __name__ == "__main__":
    main()
