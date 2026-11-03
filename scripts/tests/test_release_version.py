import json
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

    def test_hotfix_pin_survives_new_tags_through_validation_and_publication(self):
        prepared = self.version("--prepare")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertEqual(prepared.stdout.strip(), "v0.7.2")
        self.tag("v0.8.0")
        for arguments in ((), ("--prepare",)):
            result = self.version(*arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "v0.7.2")

    def test_collision_blocks_preparation_without_silently_changing_version(self):
        self.assertEqual(self.version("--prepare").returncode, 0)
        self.tag("v0.7.2")
        result = self.version("--prepare")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("v0.7.2 already exists", result.stderr)
        self.assertEqual(self.version().stdout.strip(), "v0.7.2")

    def test_unprepared_hotfix_and_state_from_another_branch_are_rejected(self):
        self.assertNotEqual(self.version().returncode, 0)
        self.assertEqual(self.version("--prepare").returncode, 0)
        result = self.version(ref="hotfix/different")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not pinned", result.stderr)

    def test_regular_release_validates_title_and_tag_before_writing_state(self):
        result = self.version(
            "--prepare", "--title", "Wrong title", ref="release/v1.0.0"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.branch / ".github/release.json").exists())
        result = self.version(
            "--prepare", "--title", "Release-1.0.0", ref="release/v1.0.0"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v1.0.0")
        result = self.version("--prepare", ref="release/v0.7.1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_malformed_metadata_cannot_supply_a_version(self):
        state = self.branch / ".github/release.json"
        state.parent.mkdir()
        for value in ([], {"branch": "hotfix/correction", "version": "wrong"}):
            with self.subTest(value=value):
                state.write_text(json.dumps(value))
                self.assertNotEqual(self.version().returncode, 0)

    def test_tag_order_is_numeric_and_prerelease_tags_are_ignored(self):
        for tag in ("v0.9.0", "v0.10.0", "v99.0.0-rc1"):
            self.tag(tag, annotated=True)
        result = self.version("--prepare")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v0.10.1")

    def test_prior_branch_metadata_is_replaced_for_a_new_hotfix(self):
        self.assertEqual(self.version("--prepare").returncode, 0)
        self.tag("v0.8.0")
        result = self.version("--prepare", ref="hotfix/different")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "v0.8.1")
        self.assertNotEqual(self.version().returncode, 0)

    def test_remote_failure_does_not_write_a_pin(self):
        self.git("remote", "set-url", "origin", str(self.root / "missing.git"))
        result = self.version("--prepare")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.branch / ".github/release.json").exists())


if __name__ == "__main__":
    unittest.main()
