import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "ci-metrics" / "publish.sh"
DATA = ".github/ci-metrics/data"

GIT = shutil.which("git")
BASH = shutil.which("bash")


@unittest.skipUnless(GIT and BASH, "needs git and bash")
class PublishScriptTests(unittest.TestCase):
    """Runs publish.sh against real local git repositories (no network)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.remote = base / "remote.git"
        self.work = base / "work"
        self.other = base / "other"

        # No git identity anywhere, like on a fresh CI runner: the script has to
        # supply its own for both the commit and the rebase. The tests' own
        # commits pass one explicitly (see commit()).
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("GIT_AUTHOR", "GIT_COMMITTER"))
        }
        self.env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            CI_METRICS_PUSH_SLEEP="0",
        )
        self.git(base, "init", "-q", "--bare", "-b", "dev", str(self.remote))
        self.git(base, "clone", "-q", str(self.remote), str(self.work))
        self.git(self.work, "checkout", "-q", "-b", "dev")
        (self.work / DATA).mkdir(parents=True)
        (self.work / DATA / ".gitkeep").write_text("")
        (self.work / "README.md").write_text("readme\n")
        self.commit(self.work, "initial")
        self.git(self.work, "push", "-q", "origin", "dev")
        self.git(base, "clone", "-q", "-b", "dev", str(self.remote), str(self.other))

    # --- helpers ---------------------------------------------------------

    def git(self, cwd, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=self.env, capture_output=True, text=True, check=check
        )

    def commit(self, repo, message):
        self.git(repo, "add", "-A")
        self.git(
            repo, "-c", "user.name=Someone", "-c", "user.email=someone@example.com", "commit", "-q", "-m", message
        )

    def publish(self, *args, repo=None, extra_env=None):
        env = {**self.env, **(extra_env or {})}
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=repo or self.work,
            env=env,
            capture_output=True,
            text=True,
        )

    def remote_files(self, ref="dev"):
        out = self.git(self.remote, "ls-tree", "-r", "--name-only", ref).stdout
        return set(out.split())

    def remote_show(self, ref, path):
        return self.git(self.remote, "show", f"{ref}:{path}").stdout

    def remote_log(self, fmt="%s"):
        return self.git(self.remote, "log", f"--format={fmt}", "dev").stdout.splitlines()

    def write_shard(self, repo, name="2026-W40.jsonl", lines=('{"a":1}', '{"a":2}')):
        (repo / DATA / name).write_text("\n".join(lines) + "\n")

    # --- tests -----------------------------------------------------------

    def test_nothing_to_publish_changes_nothing(self):
        before = self.remote_log()
        result = self.publish("dev")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("No new CI timing data", result.stdout)
        self.assertEqual(before, self.remote_log())

    def test_commits_and_pushes_new_data_with_a_record_count(self):
        self.write_shard(self.work)
        result = self.publish("dev")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(f"{DATA}/2026-W40.jsonl", self.remote_files())
        self.assertEqual('{"a":1}\n{"a":2}\n', self.remote_show("dev", f"{DATA}/2026-W40.jsonl"))
        self.assertEqual("ci: update CI timing data (+2 records)", self.remote_log()[0])

    def test_commit_uses_the_bot_identity_by_default_and_can_be_overridden(self):
        self.write_shard(self.work)
        self.assertEqual(0, self.publish("dev").returncode)
        author = self.git(self.remote, "log", "-1", "--format=%an <%ae>|%cn", "dev").stdout.strip()
        self.assertEqual(
            "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|github-actions[bot]",
            author,
        )

        self.write_shard(self.work, "2026-W41.jsonl", ('{"b":1}',))
        result = self.publish(
            "dev", extra_env={"CI_METRICS_GIT_NAME": "Metrics", "CI_METRICS_GIT_EMAIL": "m@example.com"}
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("Metrics <m@example.com>", self.git(self.remote, "log", "-1", "--format=%an <%ae>", "dev").stdout.strip())

    def test_counts_only_added_lines_when_a_shard_is_appended_to(self):
        self.write_shard(self.work)
        self.assertEqual(0, self.publish("dev").returncode)
        self.write_shard(self.work, lines=('{"a":1}', '{"a":2}', '{"a":3}', '{"a":4}', '{"a":5}'))
        self.assertEqual(0, self.publish("dev").returncode)
        self.assertEqual("ci: update CI timing data (+3 records)", self.remote_log()[0])

    def test_dry_run_reports_but_commits_and_pushes_nothing(self):
        self.write_shard(self.work)
        before = self.remote_log()
        local_before = self.git(self.work, "rev-parse", "HEAD").stdout
        result = self.publish("dev", "--dry-run")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("would commit 2 records in 1 file(s)", result.stdout)
        self.assertEqual(before, self.remote_log())
        self.assertEqual(local_before, self.git(self.work, "rev-parse", "HEAD").stdout)
        self.assertEqual("", self.git(self.work, "diff", "--cached", "--name-only").stdout)
        self.assertTrue((self.work / DATA / "2026-W40.jsonl").exists())  # the file is left alone

    def test_unknown_argument_is_rejected(self):
        self.assertEqual(2, self.publish("dev", "--force").returncode)

    def test_only_data_files_are_committed(self):
        self.write_shard(self.work)
        (self.work / "untracked.txt").write_text("not data\n")
        self.assertEqual(0, self.publish("dev").returncode)
        self.assertNotIn("untracked.txt", self.remote_files())

    def test_rebases_when_the_branch_moved_since_checkout(self):
        (self.other / "README.md").write_text("changed elsewhere\n")
        self.commit(self.other, "unrelated change")
        self.git(self.other, "push", "-q", "origin", "dev")

        self.write_shard(self.work)
        result = self.publish("dev")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("changed elsewhere\n", self.remote_show("dev", "README.md"))
        self.assertIn(f"{DATA}/2026-W40.jsonl", self.remote_files())
        self.assertEqual(["ci: update CI timing data (+2 records)", "unrelated change", "initial"], self.remote_log())

    def test_retries_a_rejected_push_and_then_succeeds(self):
        counter = Path(self.directory.name) / "rejections"
        counter.write_text("0")
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text(
            "#!/usr/bin/env bash\n"
            f"n=$(cat {counter}); if [ \"$n\" -lt 2 ]; then echo $((n+1)) > {counter}; "
            "echo 'simulated race' >&2; exit 1; fi\n"
        )
        hook.chmod(0o755)

        self.write_shard(self.work)
        result = self.publish("dev")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Push to dev was rejected (attempt 1 of 5)", result.stderr)
        self.assertIn("Push to dev was rejected (attempt 2 of 5)", result.stderr)
        self.assertIn("Pushed to dev (attempt 3)", result.stdout)
        self.assertEqual("ci: update CI timing data (+2 records)", self.remote_log()[0])

    def test_gives_up_after_five_rejected_pushes(self):
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text("#!/usr/bin/env bash\nexit 1\n")
        hook.chmod(0o755)

        self.write_shard(self.work)
        result = self.publish("dev")
        self.assertEqual(1, result.returncode)
        self.assertIn("Could not push to dev after 5 attempts", result.stderr)
        self.assertEqual(5, result.stderr.count("was rejected"))
        self.assertEqual(["initial"], self.remote_log())

    def test_conflicting_data_changes_abort_cleanly_without_pushing(self):
        (self.other / DATA / "2026-W40.jsonl").write_text('{"other":1}\n')
        self.commit(self.other, "someone else wrote this shard")
        self.git(self.other, "push", "-q", "origin", "dev")

        self.write_shard(self.work)
        result = self.publish("dev")
        self.assertEqual(1, result.returncode)
        self.assertIn("Could not rebase onto origin/dev", result.stderr)
        self.assertEqual(["someone else wrote this shard", "initial"], self.remote_log())
        status = self.git(self.work, "status", "--porcelain").stdout
        self.assertNotIn("UU", status)
        self.assertFalse((self.work / ".git" / "rebase-merge").exists())

    def test_refuses_to_push_anything_outside_the_data_directory(self):
        self.git(self.work, "checkout", "-q", "-b", "feature")
        (self.work / "README.md").write_text("feature work\n")
        self.commit(self.work, "feature commit")
        self.write_shard(self.work)

        result = self.publish("dev")
        self.assertEqual(1, result.returncode)
        self.assertIn("Refusing to push", result.stderr)
        self.assertIn("README.md", result.stderr)
        self.assertEqual(["initial"], self.remote_log())

    def test_works_on_a_branch_other_than_dev(self):
        self.git(self.work, "push", "-q", "origin", "dev:staging")
        self.git(self.work, "checkout", "-q", "-B", "staging", "origin/staging")
        self.write_shard(self.work)
        result = self.publish("staging")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(f"{DATA}/2026-W40.jsonl", self.remote_files("staging"))
        self.assertNotIn(f"{DATA}/2026-W40.jsonl", self.remote_files("dev"))


if __name__ == "__main__":
    unittest.main()
