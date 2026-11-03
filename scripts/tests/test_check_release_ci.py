import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from urllib.parse import parse_qs, urlsplit


SCRIPT = Path(__file__).resolve().parents[1] / "check_release_ci.py"
SHA = "a" * 40


def _load_workflows():
    spec = importlib.util.spec_from_file_location("check_release_ci", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORKFLOWS


WORKFLOWS = _load_workflows()
TARGET_WORKFLOW = WORKFLOWS[-1]
OTHER_WORKFLOWS = WORKFLOWS[:-1]


class ReleaseCITests(unittest.TestCase):
    def setUp(self):
        self.responses = {}
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                request = urlsplit(self.path)
                owner.requests.append((request.path, parse_qs(request.query)))
                response = owner.responses.get(request.path.split("/")[-2], [])
                if isinstance(response, deque):
                    response = response.popleft() if len(response) > 1 else response[0]
                self.send_response(403 if response == "forbidden" else 200)
                self.end_headers()
                self.wfile.write(
                    response
                    if isinstance(response, bytes)
                    else json.dumps({"workflow_runs": response}).encode()
                )

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)

    def run_gate(self, timeout="0"):
        environment = {
            **os.environ,
            "GITHUB_API_URL": f"http://127.0.0.1:{self.server.server_port}",
        }
        environment.pop("GH_TOKEN", None)
        environment.pop("GITHUB_TOKEN", None)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "MolarVerse/PQ",
                SHA,
                "--timeout",
                timeout,
                "--interval",
                "0.01",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )

    @staticmethod
    def workflow_run(**changes):
        return {
            "id": 10,
            "head_sha": SHA,
            "head_branch": "main",
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            **changes,
        }

    def test_requires_all_post_merge_workflows_for_the_exact_commit(self):
        for workflow in WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.requests), len(WORKFLOWS))
        for path, query in self.requests:
            self.assertTrue(path.startswith("/repos/MolarVerse/PQ/actions/workflows/"))
            self.assertEqual(
                query,
                {
                    "head_sha": [SHA],
                    "branch": ["main"],
                    "event": ["push"],
                    "per_page": ["100"],
                },
            )

    def test_failure_cancellation_and_skipped_ci_block_publication(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        for conclusion in (
            "failure",
            "cancelled",
            "timed_out",
            "skipped",
            "neutral",
            "action_required",
        ):
            with self.subTest(conclusion=conclusion):
                self.responses[TARGET_WORKFLOW] = [
                    self.workflow_run(conclusion=conclusion)
                ]
                result = self.run_gate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(conclusion, result.stderr)

    def test_other_commits_branches_and_pr_checks_cannot_satisfy_gate(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        for changes in (
            {"head_sha": "b" * 40},
            {"head_branch": "dev"},
            {"event": "pull_request"},
        ):
            with self.subTest(changes=changes):
                self.responses[TARGET_WORKFLOW] = [self.workflow_run(**changes)]
                result = self.run_gate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"still waiting for {TARGET_WORKFLOW}", result.stderr)

    def test_missing_and_pending_runs_fail_closed_at_timeout(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        for runs in ([], [self.workflow_run(status="in_progress", conclusion=None)]):
            with self.subTest(runs=runs):
                self.responses[TARGET_WORKFLOW] = runs
                self.assertNotEqual(self.run_gate().returncode, 0)

    def test_latest_run_wins_over_an_older_success(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        for newer in (
            self.workflow_run(id=11, conclusion="failure"),
            self.workflow_run(id=11, status="queued", conclusion=None),
        ):
            with self.subTest(newer=newer):
                self.responses[TARGET_WORKFLOW] = [newer, self.workflow_run()]
                self.assertNotEqual(self.run_gate().returncode, 0)
        self.responses[TARGET_WORKFLOW] = [
            self.workflow_run(id=9, conclusion="failure"),
            self.workflow_run(),
        ]
        self.assertEqual(self.run_gate().returncode, 0)

    def test_waits_for_post_merge_ci_to_finish(self):
        iterations = 3
        self.responses[TARGET_WORKFLOW] = deque(
            [
                [],
                [self.workflow_run(status="in_progress", conclusion=None)],
                [self.workflow_run()],
            ]
        )
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        result = self.run_gate(timeout="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Waiting for", result.stdout)
        self.assertEqual(len(self.requests), iterations * len(WORKFLOWS))

    def test_api_permission_errors_block_publication(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        self.responses[TARGET_WORKFLOW] = "forbidden"
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HTTP 403", result.stderr)

    def test_malformed_api_data_blocks_publication(self):
        for workflow in OTHER_WORKFLOWS:
            self.responses[workflow] = [self.workflow_run()]
        for body in (b"{broken", b"null", b"[]", b"{}", b'{"workflow_runs":[{}]}'):
            with self.subTest(body=body):
                self.responses[TARGET_WORKFLOW] = body
                result = self.run_gate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Cannot verify release CI", result.stderr)


if __name__ == "__main__":
    unittest.main()
