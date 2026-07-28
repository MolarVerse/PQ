import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "check_changelog_fragment.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "check_changelog_fragment", SCRIPT
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ChangelogFragmentCheckTests(unittest.TestCase):
    def test_accepts_exactly_one_valid_fragment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "workflow.developer.ci.md"
            path.parent.mkdir()
            path.write_text("- Enforce changelog audiences.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [("A", "changes/workflow.developer.ci.md")], root
            )

            self.assertEqual([], errors)

    def test_rejects_missing_or_multiple_fragments(self):
        self.assertTrue(CHECK.validate_pr_changes([]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changes = root / "changes"
            changes.mkdir()
            first = changes / "first.user.bugfix.md"
            second = changes / "second.developer.test.md"
            first.write_text("- Fix the output.\n", encoding="utf-8")
            second.write_text("- Cover the output.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [
                    ("A", "changes/first.user.bugfix.md"),
                    ("A", "changes/second.developer.test.md"),
                ],
                root,
            )

            self.assertTrue(
                any("exactly one" in error for error in errors)
            )

    def test_rejects_direct_changelog_edits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "workflow.developer.ci.md"
            path.parent.mkdir()
            path.write_text("- Enforce changelog audiences.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [
                    ("M", "CHANGELOG.md"),
                    ("A", "changes/workflow.developer.ci.md"),
                ],
                root,
            )

            self.assertTrue(
                any("must not edit" in error for error in errors)
            )

    def test_rejects_modified_existing_fragments(self):
        errors = CHECK.validate_pr_changes(
            [("M", "changes/existing.developer.internal.md")]
        )

        self.assertTrue(
            any("immutable" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
