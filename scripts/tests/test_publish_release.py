import json
from pathlib import Path
import subprocess
import sys
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Lock, Thread
import unittest

from release_git_fixture import LocalReleaseRepository


SCRIPT = Path(__file__).resolve().parents[1] / "publish_release.py"
VERSION = "v9.9.9"


class PublicationTests(LocalReleaseRepository, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.release = None
        self.get_status = None
        self.get_body = None
        self.post_mode = "success"
        self.posts = []
        self.requests = []
        self.after_post = None
        self.before_post = None
        self.release_lock = Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def respond(self, status, data):
                self.send_response(status)
                self.end_headers()
                self.wfile.write(
                    data if isinstance(data, bytes) else json.dumps(data).encode()
                )

            def do_GET(self):
                owner.requests.append(self.path)
                if self.path != f"/repos/MolarVerse/PQ/releases/tags/{VERSION}":
                    self.respond(404, {})
                elif owner.get_status:
                    self.respond(owner.get_status, {})
                elif owner.get_body is not None:
                    self.respond(200, owner.get_body)
                elif owner.release is None:
                    self.respond(404, {})
                else:
                    self.respond(200, owner.release)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.posts.append(body)
                if self.path != "/repos/MolarVerse/PQ/releases":
                    self.respond(404, {})
                    return
                if owner.before_post:
                    owner.before_post()
                with owner.release_lock:
                    if owner.release is not None:
                        self.respond(422, {"message": "release already exists"})
                        return
                    owner.release = {**body, "published_at": "2026-01-01T00:00:00Z"}
                if owner.after_post:
                    owner.after_post()
                if owner.post_mode == "error_after_write":
                    self.respond(503, {})
                elif owner.post_mode == "empty_after_write":
                    self.respond(201, b"")
                elif owner.post_mode == "disconnect_after_write":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                else:
                    self.respond(201, owner.release)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def command(self):
        return [sys.executable, str(SCRIPT), "MolarVerse/PQ", VERSION]

    def environment(self):
        env = {
            **self.git_environment,
            "GITHUB_API_URL": f"http://127.0.0.1:{self.server.server_port}",
        }
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        return env

    def publish(self):
        return subprocess.run(
            self.command(),
            cwd=self.branch,
            env=self.environment(),
            capture_output=True,
            text=True,
            timeout=5,
        )

    def published(self, **changes):
        return {
            "tag_name": VERSION,
            "draft": False,
            "prerelease": False,
            "published_at": "2026-01-01T00:00:00Z",
            **changes,
        }

    def remote_tags(self):
        return self.git(
            "for-each-ref", "--format=%(objectname)", "refs/tags/", remote=True
        ).stdout.strip()

    def test_creates_exact_tag_and_stable_release_then_retries_without_duplicate_post(
        self,
    ):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.remote_tags(), self.commit)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["target_commitish"], self.commit)
        self.assertEqual(self.posts[0]["tag_name"], VERSION)
        self.assertIs(self.posts[0]["draft"], False)
        self.assertIs(self.posts[0]["prerelease"], False)
        self.assertIs(self.posts[0]["generate_release_notes"], True)
        self.assertEqual(self.publish().returncode, 0)
        self.assertEqual(len(self.posts), 1)

    def test_existing_annotated_tag_is_accepted_without_replacing_it(self):
        self.tag(VERSION, annotated=True)
        original = self.remote_tags()
        self.assertNotEqual(original, self.commit)
        self.release = self.published()
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.remote_tags(), original)
        self.assertEqual(self.posts, [])

    def test_remote_or_local_tags_for_other_commits_are_never_replaced(self):
        other = self.git(
            "commit-tree", self.tree, "-p", self.commit, input="other\n"
        ).stdout.strip()
        self.git("tag", VERSION, other)
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local tag", result.stderr)
        self.assertEqual(self.remote_tags(), "")
        self.tag(VERSION, commit=other)
        self.requests.clear()
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another commit", result.stderr)
        self.assertEqual(self.remote_tags(), other)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.posts, [])

    def test_drafts_prereleases_and_incomplete_records_are_not_reported_as_published(
        self,
    ):
        self.tag(VERSION)
        for changes in (
            {"draft": True},
            {"prerelease": True},
            {"published_at": None},
            {"tag_name": "v0.0.0"},
        ):
            with self.subTest(changes=changes):
                self.release = self.published(**changes)
                result = self.publish()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not a published stable release", result.stderr)
                self.assertEqual(self.remote_tags(), self.commit)
                self.assertEqual(self.posts, [])

    def test_existing_release_without_its_tag_requires_manual_recovery(self):
        self.release = self.published()
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has no tag", result.stderr)
        self.assertEqual(self.remote_tags(), "")
        self.assertEqual(self.posts, [])

    def test_api_outages_and_invalid_responses_do_not_create_tags_or_releases(self):
        for status, body in (
            (403, None),
            (500, None),
            (None, b"{broken"),
            (None, b"null"),
            (None, b"[]"),
        ):
            with self.subTest(status=status, body=body):
                self.get_status, self.get_body = status, body
                result = self.publish()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.remote_tags(), "")
                self.assertEqual(self.posts, [])

    def test_unknown_create_outcome_is_verified_on_retry_without_reposting(self):
        for mode in (
            "error_after_write",
            "empty_after_write",
            "disconnect_after_write",
        ):
            with self.subTest(mode=mode):
                self.post_mode = mode
                self.release = None
                self.posts.clear()
                first = self.publish()
                self.assertNotEqual(first.returncode, 0)
                self.assertEqual(self.remote_tags(), self.commit)
                self.assertEqual(len(self.posts), 1)
                second = self.publish()
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertEqual(len(self.posts), 1)

    def test_local_annotated_tag_is_pushed_without_replacing_the_annotation(self):
        self.git("tag", "-a", VERSION, self.commit, "-m", "original annotation")
        original = self.git("rev-parse", f"refs/tags/{VERSION}").stdout.strip()
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.remote_tags(), original)
        self.assertEqual(self.posts[0]["target_commitish"], self.commit)

    def test_simultaneous_same_commit_requests_have_a_recoverable_duplicate_conflict(
        self,
    ):
        self.tag(VERSION)
        barrier = Barrier(2, timeout=3)
        self.before_post = barrier.wait
        processes = [
            subprocess.Popen(
                self.command(),
                cwd=self.branch,
                env=self.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        outputs = [process.communicate(timeout=5) for process in processes]
        self.assertEqual(
            sorted(process.returncode for process in processes), [0, 1], outputs
        )
        self.assertEqual(len(self.posts), 2)
        self.assertEqual(self.remote_tags(), self.commit)
        self.assertEqual(self.release["target_commitish"], self.commit)
        self.before_post = None
        retry = self.publish()
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertEqual(len(self.posts), 2)

    def test_rejected_tag_push_can_be_retried_with_the_same_local_tag(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        first = self.publish()
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self.remote_tags(), "")
        self.assertEqual(self.posts, [])
        hook.unlink()
        second = self.publish()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.remote_tags(), self.commit)
        self.assertEqual(len(self.posts), 1)

    def test_unreachable_remote_stops_before_contacting_release_api(self):
        self.git("remote", "set-url", "origin", str(self.root / "missing.git"))
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.posts, [])

    def test_concurrent_different_commits_cannot_publish_the_same_version(self):
        second_branch = self.root / "second"
        subprocess.run(
            ["git", "clone", str(self.remote), str(second_branch)],
            env=self.git_environment,
            check=True,
            capture_output=True,
        )
        other = self.git(
            "commit-tree", self.tree, "-p", self.commit, input="other\n"
        ).stdout.strip()
        self.git("push", str(second_branch), f"{other}:refs/heads/other")
        subprocess.run(
            ["git", "checkout", "other"],
            cwd=second_branch,
            env=self.git_environment,
            check=True,
            capture_output=True,
        )
        processes = [
            subprocess.Popen(
                self.command(),
                cwd=branch,
                env=self.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for branch in (self.branch, second_branch)
        ]
        for process in processes:
            process.communicate(timeout=5)
        self.assertEqual(sorted(process.returncode for process in processes), [0, 1])
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["target_commitish"], self.remote_tags())

    def test_tag_moved_during_creation_is_not_reported_as_success(self):
        other = self.git(
            "commit-tree", self.tree, "-p", self.commit, input="other\n"
        ).stdout.strip()
        self.git("push", "origin", f"{other}:refs/heads/other")
        self.after_post = lambda: self.git(
            "update-ref", f"refs/tags/{VERSION}", other, remote=True
        )
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Verified published release", result.stdout)


if __name__ == "__main__":
    unittest.main()
