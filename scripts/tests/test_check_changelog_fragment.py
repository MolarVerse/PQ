import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
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
    def test_accepts_one_valid_fragment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "developer" / "ci.workflow.md"
            path.parent.mkdir(parents=True)
            path.write_text("- Enforce changelog audiences.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [("A", "changes/developer/ci.workflow.md")], root
            )

            self.assertEqual([], errors)

    def test_requires_at_least_one_fragment(self):
        self.assertTrue(CHECK.validate_pr_changes([]))

    def test_accepts_multiple_fragments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changes = root / "changes"
            (changes / "user").mkdir(parents=True)
            (changes / "developer").mkdir(parents=True)
            first = changes / "user" / "bugfix.first.md"
            second = changes / "developer" / "test.second.md"
            first.write_text("- Fix the output.\n", encoding="utf-8")
            second.write_text("- Cover the output.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [
                    ("A", "changes/user/bugfix.first.md"),
                    ("A", "changes/developer/test.second.md"),
                ],
                root,
            )

            self.assertEqual([], errors)

    def test_accepts_fragment_with_multiple_bullets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "user" / "enhancement.batch.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "- First point.\n- Second point.\n", encoding="utf-8"
            )

            errors = CHECK.validate_pr_changes(
                [("A", "changes/user/enhancement.batch.md")], root
            )

            self.assertEqual([], errors)

    def test_validates_every_changed_fragment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changes = root / "changes" / "user"
            changes.mkdir(parents=True)
            valid = changes / "bugfix.first.md"
            invalid = changes / "ci.second.md"
            valid.write_text("- Fix the output.\n", encoding="utf-8")
            invalid.write_text("- Cover the output.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [
                    ("A", "changes/user/bugfix.first.md"),
                    ("A", "changes/user/ci.second.md"),
                ],
                root,
            )

            self.assertTrue(
                any("invalid user category 'ci'" in error for error in errors)
            )

    def test_invalid_name_error_explains_allowed_characters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "user" / "bugfix.Slug-nAme.md"
            path.parent.mkdir(parents=True)
            path.write_text("- Fix the output.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [("A", "changes/user/bugfix.Slug-nAme.md")], root
            )

            self.assertEqual(1, len(errors))
            self.assertIn("uses only lowercase letters", errors[0])
            self.assertIn(
                "uses only lowercase letters, digits, and hyphens", errors[0]
            )

    def test_rejects_direct_changelog_edits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "developer" / "ci.workflow.md"
            path.parent.mkdir(parents=True)
            path.write_text("- Enforce changelog audiences.\n", encoding="utf-8")

            errors = CHECK.validate_pr_changes(
                [
                    ("M", "CHANGELOG.md"),
                    ("A", "changes/developer/ci.workflow.md"),
                ],
                root,
            )

            self.assertTrue(
                any("must not edit" in error for error in errors)
            )

    def test_accepts_modified_existing_fragments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "changes" / "developer" / "internal.existing.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                "- Correct an unreleased changelog entry.\n"
                "- Add a second point to the same entry.\n",
                encoding="utf-8",
            )

            errors = CHECK.validate_pr_changes(
                [("M", "changes/developer/internal.existing.md")], root
            )

            self.assertEqual([], errors)

    def test_rejects_deleted_existing_fragments(self):
        errors = CHECK.validate_pr_changes(
            [("D", "changes/developer/internal.existing.md")]
        )

        self.assertTrue(any("must not delete" in error for error in errors))

    def test_accepts_entry_preserving_fragment_reorganization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changes = root / "changes" / "developer"
            changes.mkdir(parents=True)
            (changes / "test.first.md").write_text(
                "- First test entry.\n", encoding="utf-8"
            )
            (changes / "test.second.md").write_text(
                "- Second test entry.\n", encoding="utf-8"
            )

            errors = CHECK.validate_pr_changes(
                [
                    ("D", "changes/developer/test.pre-fragment-backlog.md"),
                    ("A", "changes/developer/test.first.md"),
                    ("A", "changes/developer/test.second.md"),
                ],
                root,
                Counter({("developer", "Tests"): 2}),
            )

            self.assertEqual([], errors)

    def test_rejects_fragment_reorganization_that_loses_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            changes = root / "changes" / "developer"
            changes.mkdir(parents=True)
            (changes / "test.first.md").write_text(
                "- Only one entry remains.\n", encoding="utf-8"
            )

            errors = CHECK.validate_pr_changes(
                [
                    ("D", "changes/developer/test.pre-fragment-backlog.md"),
                    ("A", "changes/developer/test.first.md"),
                ],
                root,
                Counter({("developer", "Tests"): 2}),
            )

            self.assertTrue(
                any("missing developer/Tests: 1" in error for error in errors)
            )

    def test_ignores_unrelated_changes_directory_paths(self):
        errors = CHECK.validate_pr_changes(
            [("A", "changes/README.md"), ("A", "changes/user/README.md")]
        )

        self.assertTrue(
            any(
                "must add or update at least one" in error
                for error in errors
            )
        )


if __name__ == "__main__":
    unittest.main()
