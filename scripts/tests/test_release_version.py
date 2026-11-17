from pathlib import Path
import subprocess
import sys
import unittest
from release_git_fixture import LocalReleaseRepository


SCRIPT = Path(__file__).resolve().parents[1] / "release_version.py"


class ReleaseVersionTests(LocalReleaseRepository, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tag("v0.7.1")

    def version(self, *arguments, ref="hotfix/correction"):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--head-ref", ref, *arguments],
            cwd=self.branch,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_hotfix_version_comes_from_title_and_survives_new_tags(self):
        title = ("--title", "Hotfix-0.7.2: correct a thing")
        prepared = self.version("--prepare", *title)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(prepared.stdout.strip(), "v0.7.2")
        self.assertFalse((self.branch / ".github/release.json").exists())
        self.tag("v0.8.0")
        result = self.version(*title)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v0.7.2")

    def test_hotfix_title_must_carry_a_version(self):
        for title in (None, "fix: correct a thing", "Hotfix-0.7", "Hotfix-0.7.2x"):
            with self.subTest(title=title):
                arguments = () if title is None else ("--title", title)
                for prepare in ((), ("--prepare",)):
                    result = self.version(*prepare, *arguments)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Hotfix-X.Y.Z", result.stderr)

    def test_hotfix_version_must_follow_the_latest_tag(self):
        for title in ("Hotfix-0.7.3", "Hotfix-0.8.0", "Hotfix-0.7.1"):
            with self.subTest(title=title):
                result = self.version("--prepare", "--title", title)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be v0.7.2", result.stderr)

    def test_collision_blocks_preparation(self):
        self.tag("v0.7.2")
        result = self.version("--prepare", "--title", "Hotfix-0.7.2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be v0.7.3", result.stderr)

    def test_regular_release_validates_title_and_tag(self):
        result = self.version(
            "--prepare", "--title", "Wrong title", ref="release/v1.0.0"
        )
        self.assertNotEqual(result.returncode, 0)
        result = self.version(
            "--prepare", "--title", "Release-1.0.0", ref="release/v1.0.0"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v1.0.0")
        result = self.version("--prepare", ref="release/v0.7.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_tag_order_is_numeric_and_prerelease_tags_are_ignored(self):
        for tag in ("v0.9.0", "v0.10.0", "v99.0.0-rc1"):
            self.tag(tag, annotated=True)
        result = self.version("--prepare", "--title", "Hotfix-0.10.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v0.10.1")

    def test_remote_failure_blocks_preparation(self):
        self.git("remote", "set-url", "origin", str(self.root / "missing.git"))
        result = self.version("--prepare", "--title", "Hotfix-0.7.2")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
