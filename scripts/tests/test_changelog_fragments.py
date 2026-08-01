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
            "Bug Fixes",
            fragments.parse_fragment_name("bugfix.trajectory.md", "user"),
        )
        self.assertEqual(
            "CI",
            fragments.parse_fragment_name("ci.changelog.md", "developer"),
        )

    def test_category_must_match_audience(self):
        with self.assertRaisesRegex(
            fragments.FragmentError, "invalid user category 'ci'"
        ):
            fragments.parse_fragment_name("ci.workflow.md", "user")

    def test_name_must_match_category_dot_slug(self):
        with self.assertRaisesRegex(fragments.FragmentError, "invalid fragment name"):
            fragments.parse_fragment_name("trajectory.user.bugfix.md", "user")

    def test_entry_must_be_one_or_more_bullets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "change.entry.md"

            path.write_text("Not a bullet.\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fragments.FragmentError, "Markdown bullets"
            ):
                fragments.read_fragment_entries(path)

            path.write_text("- First.\n\n- Second.\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fragments.FragmentError, "Markdown bullets"
            ):
                fragments.read_fragment_entries(path)

            path.write_text("- One concise entry.\n", encoding="utf-8")
            self.assertEqual(
                ["- One concise entry."],
                fragments.read_fragment_entries(path),
            )

            path.write_text(
                "- First point.\n- Second point.\n", encoding="utf-8"
            )
            self.assertEqual(
                ["- First point.", "- Second point."],
                fragments.read_fragment_entries(path),
            )

    def test_entry_must_end_with_newline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "change.entry.md"
            path.write_text("- No trailing newline.", encoding="utf-8")

            with self.assertRaisesRegex(
                fragments.FragmentError, "must end with a newline"
            ):
                fragments.read_fragment_entries(path)

    def test_load_fragments_reads_both_audience_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            changes = Path(temporary_directory)
            (changes / "user").mkdir()
            (changes / "developer").mkdir()

            (changes / "user" / "bugfix.trajectory.md").write_text(
                "- Fix trajectory output.\n", encoding="utf-8"
            )
            (changes / "developer" / "test.parser.md").write_text(
                "- Cover parser edge cases.\n- Add regression fixture.\n",
                encoding="utf-8",
            )
            (changes / "user" / "README.md").write_text(
                "ignored\n", encoding="utf-8"
            )

            loaded = fragments.load_fragments(changes)

            self.assertEqual(2, len(loaded))
            by_audience = {fragment.audience: fragment for fragment in loaded}
            self.assertEqual("Bug Fixes", by_audience["user"].section)
            self.assertEqual(
                ["- Fix trajectory output."], by_audience["user"].entries
            )
            self.assertEqual("Tests", by_audience["developer"].section)
            self.assertEqual(
                ["- Cover parser edge cases.", "- Add regression fixture."],
                by_audience["developer"].entries,
            )


if __name__ == "__main__":
    unittest.main()
