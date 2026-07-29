import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import changelog_fragments as fragments


class ChangelogFragmentTests(unittest.TestCase):
    def test_user_and_developer_names_select_distinct_sections(self):
        self.assertEqual(
            ("user", "Bug Fixes"),
            fragments.parse_fragment_name("trajectory.user.bugfix.md"),
        )
        self.assertEqual(
            ("developer", "CI"),
            fragments.parse_fragment_name("changelog.developer.ci.md"),
        )

    def test_category_must_match_audience(self):
        with self.assertRaisesRegex(
            fragments.FragmentError, "invalid user category 'ci'"
        ):
            fragments.parse_fragment_name("workflow.user.ci.md")

    def test_entry_must_be_one_concise_bullet(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "entry.user.change.md"

            path.write_text("Not a bullet.\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fragments.FragmentError, "exactly one"
            ):
                fragments.read_fragment_entry(path)

            path.write_text("- First.\n- Second.\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fragments.FragmentError, "exactly one"
            ):
                fragments.read_fragment_entry(path)

            path.write_text("- One concise entry.\n", encoding="utf-8")
            self.assertEqual(
                "- One concise entry.",
                fragments.read_fragment_entry(path),
            )

    def test_legacy_fragments_remain_release_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            changes = Path(temporary_directory)
            path = changes / "old-branch.test.md"
            path.write_text("- Cover the old branch.\n", encoding="utf-8")

            loaded = fragments.load_fragments(changes)

            self.assertEqual(1, len(loaded))
            self.assertEqual("developer", loaded[0].audience)
            self.assertEqual("Tests", loaded[0].section)


if __name__ == "__main__":
    unittest.main()
