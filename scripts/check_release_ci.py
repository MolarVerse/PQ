#!/usr/bin/env python3
"""Wait for successful post-merge BUILD and LINT runs on a release commit."""

import argparse
import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WORKFLOWS = ("ci_build.yml")


def latest_run(api_url, repository, workflow, sha):
    query = urlencode(
        {
            "head_sha": sha,
            "branch": "main",
            "event": "push",
            "per_page": 100,
        }
    )
    url = f"{api_url}/repos/{repository}/actions/workflows/{workflow}/runs?{query}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        runs = json.load(response)["workflow_runs"]
    matching = [
        run
        for run in runs
        if run["head_sha"] == sha
        and run["head_branch"] == "main"
        and run["event"] == "push"
    ]
    return max(matching, key=lambda run: run["id"], default=None)


def wait_for_ci(api_url, repository, sha, timeout, interval):
    deadline = time.monotonic() + timeout
    while True:
        pending = []
        for workflow in WORKFLOWS:
            run = latest_run(api_url, repository, workflow, sha)
            if run is None or run["status"] != "completed":
                pending.append(workflow)
            elif run["conclusion"] != "success":
                raise ValueError(
                    f"{workflow} did not succeed for {sha}: "
                    f"{run['conclusion']} ({run.get('html_url', '')})"
                )
        if not pending:
            print(f"BUILD and LINT succeeded on main at {sha}.", flush=True)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"No successful CI for {sha}; still waiting for {', '.join(pending)}"
            )
        print(f"Waiting for {sha}: {', '.join(pending)}", flush=True)
        time.sleep(min(interval, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("sha")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--interval", type=float, default=20)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("repository must use owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("publication requires a full commit SHA")
    if args.timeout < 0 or args.interval <= 0:
        parser.error("timeout must be non-negative and interval must be positive")
    try:
        wait_for_ci(
            os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
            args.repository,
            args.sha,
            args.timeout,
            args.interval,
        )
    except HTTPError as error:
        sys.exit(f"Cannot verify release CI: GitHub returned HTTP {error.code}")
    except (URLError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"Cannot verify release CI: {error}")


if __name__ == "__main__":
    main()
