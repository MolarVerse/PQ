"""Tests of .githooks/commit-msg: it is run on a message file, as git does."""

import subprocess
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "commit-msg"


def run_hook(message):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "COMMIT_EDITMSG"
        path.write_text(message)
        return subprocess.run(["bash", str(HOOK), str(path)], capture_output=True, text=True, cwd=tmp).returncode


class CommitMsgHookTest(unittest.TestCase):
    def test_conventional_prefixes_are_accepted(self):
        for message in ("feat: add x", "fix(scope): y", "docs: z", "ci: w", "merge: dev into feature/x", "test: t"):
            with self.subTest(message=message):
                self.assertEqual(0, run_hook(message + "\n"))

    def test_messages_without_a_prefix_are_rejected(self):
        for message in ("add x", "Update the readme", "merging things", "feature add x"):
            with self.subTest(message=message):
                self.assertEqual(1, run_hook(message + "\n"))

    def test_the_merge_messages_git_proposes_are_accepted(self):
        for message in (
            "Merge branch 'dev' into feature/x",
            "Merge branch 'feature/x' of github.com:MolarVerse/PQ into feature/x",
            "Merge branches 'a' and 'b'",
            "Merge remote-tracking branch 'origin/dev' into feature/x",
            "Merge tag 'v1.0.0' into dev",
            "Merge commit 'abc123' into dev",
            "Merge pull request #12 from MolarVerse/feature/x",
        ):
            with self.subTest(message=message):
                self.assertEqual(0, run_hook(message + "\n\n# Conflicts:\n#\tfile\n"))

    def test_only_git_merge_messages_are_exempt(self):
        for message in ("Merge", "Merge it", "Merge stuff into dev", "Merging branch x", "Mergebranch 'x'"):
            with self.subTest(message=message):
                self.assertEqual(1, run_hook(message + "\n"))

    def test_the_message_file_argument_is_used(self):
        # run from a directory without .git: the hook must not depend on .git/COMMIT_EDITMSG
        self.assertEqual(0, run_hook("fix: a\n"))


if __name__ == "__main__":
    unittest.main()
