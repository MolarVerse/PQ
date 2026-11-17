from pathlib import Path
import subprocess
import sys
import unittest

from release_git_fixture import LocalReleaseRepository


SCRIPT = Path(__file__).resolve().parents[1] / "check_release_refs.py"


class ReleaseRefTests(LocalReleaseRepository, unittest.TestCase):
    def availability(self, version="v9.9.9"):
        return subprocess.run(
            [sys.executable, str(SCRIPT), version],
            cwd=self.branch,
            env=self.git_environment,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_available_refs_pass_and_existing_release_tag_or_branch_blocks(self):
        self.assertEqual(self.availability().returncode, 0)
        self.tag("v9.9.9", annotated=True)
        result = self.availability()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.git("push", "origin", f"{self.commit}:refs/heads/release/v9.9.8")
        self.assertNotEqual(self.availability("v9.9.8").returncode, 0)
        self.assertEqual(self.availability("v9.9.7").returncode, 0)

    def test_remote_error_is_never_reported_as_available(self):
        self.git("remote", "set-url", "origin", str(self.root / "missing.git"))
        result = self.availability()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot check release refs", result.stderr)
        self.assertNotIn("are available", result.stdout)

    def test_invalid_version_stops_before_querying_remote(self):
        self.git("remote", "remove", "origin")
        result = self.availability("../invalid")
        self.assertEqual(result.returncode, 2)
        self.assertIn("version must use", result.stderr)


if __name__ == "__main__":
    unittest.main()
