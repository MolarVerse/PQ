import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "update_changelog.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("update_changelog", SCRIPT)
CHANGELOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHANGELOG)


class ChangelogTests(unittest.TestCase):
    def test_release_notes_require_a_bullet(self):
        self.assertFalse(
            CHANGELOG.has_release_notes(
                ["", "### Bug Fixes", "", "No release notes yet."]
            )
        )
        self.assertTrue(
            CHANGELOG.has_release_notes(
                ["", "### Bug Fixes", "", "- Fix trajectory output."]
            )
        )

    def test_stamp_keeps_empty_next_release_and_history(self):
        lines = [
            "# Changelog",
            "",
            "## Next Release",
            "",
            "### Bug Fixes",
            "",
            "- Fix trajectory output.",
            "",
            "<!-- insertion marker -->",
            "## [v1.0.0](release-url) - 2025-01-01",
        ]

        head, body, tail = CHANGELOG.split_next_release(
            lines, "CHANGELOG.md"
        )
        stamped = CHANGELOG.stamp_release(
            head, body, tail, "v1.1.0", "MolarVerse/PQ"
        )

        next_index = stamped.index("## Next Release")
        marker_index = stamped.index("<!-- insertion marker -->")
        release_index = next(
            index
            for index, line in enumerate(stamped)
            if line.startswith("## [v1.1.0]")
        )
        old_release_index = stamped.index(
            "## [v1.0.0](release-url) - 2025-01-01"
        )

        self.assertLess(next_index, marker_index)
        self.assertLess(marker_index, release_index)
        self.assertLess(release_index, old_release_index)
        self.assertIn(
            "- Fix trajectory output.",
            stamped[release_index:old_release_index],
        )
        self.assertFalse(
            CHANGELOG.has_release_notes(
                stamped[next_index + 1 : marker_index]
            )
        )

    def test_release_routes_fragments_and_preserves_unreleased_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            user_changelog = root / "CHANGELOG.md"
            dev_changelog = root / "DEV-CHANGELOG.md"
            changes_dir = root / "changes"
            changes_dir.mkdir()

            user_changelog.write_text(
                "# Changelog\n\n"
                "## Next Release\n\n"
                "### Bug Fixes\n\n"
                "- Fix trajectory output.\n\n"
                "<!-- insertion marker -->\n"
                "## [v1.0.0](release-url) - 2025-01-01\n",
                encoding="utf-8",
            )
            dev_changelog.write_text(
                "# Developer Changelog\n\n"
                "## Next Release\n\n"
                "### Internal\n\n"
                "- Existing developer note.\n\n"
                "<!-- insertion marker -->\n"
                "## [v1.0.0](release-url) - 2025-01-01\n",
                encoding="utf-8",
            )
            (changes_dir / "user").mkdir()
            (changes_dir / "developer").mkdir()
            user_fragment = (
                changes_dir / "user" / "enhancement.trajectory.md"
            )
            user_fragment.write_text(
                "- Add effective step metadata to trajectory output.\n",
                encoding="utf-8",
            )
            developer_fragment = (
                changes_dir / "developer" / "test.parser.md"
            )
            developer_fragment.write_text(
                "- Cover changelog fragment parsing.\n"
                "- Add a regression fixture.\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    CHANGELOG, "USER_CHANGELOG", user_changelog
                ),
                mock.patch.object(
                    CHANGELOG, "DEV_CHANGELOG", dev_changelog
                ),
                mock.patch.object(CHANGELOG, "CHANGES_DIR", changes_dir),
            ):
                CHANGELOG.update_changelogs("v1.1.0")

            user_text = user_changelog.read_text(encoding="utf-8")
            dev_text = dev_changelog.read_text(encoding="utf-8")

            self.assertIn("## Next Release\n\n<!-- insertion marker -->", user_text)
            self.assertIn("- Fix trajectory output.", user_text)
            self.assertIn(
                "- Add effective step metadata to trajectory output.",
                user_text,
            )
            self.assertNotIn("Cover changelog fragment parsing", user_text)
            self.assertIn("- Existing developer note.", dev_text)
            self.assertIn("- Cover changelog fragment parsing.", dev_text)
            self.assertIn("- Add a regression fixture.", dev_text)
            self.assertNotIn("effective step metadata", dev_text)
            self.assertFalse(user_fragment.exists())
            self.assertFalse(developer_fragment.exists())

    def test_developer_only_release_leaves_user_changelog_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            user_changelog = root / "CHANGELOG.md"
            dev_changelog = root / "DEV-CHANGELOG.md"
            changes_dir = root / "changes"
            changes_dir.mkdir()

            user_text = (
                "# Changelog\n\n"
                "## Next Release\n\n"
                "<!-- insertion marker -->\n"
                "## [v1.0.0](release-url) - 2025-01-01\n"
            )
            user_changelog.write_text(user_text, encoding="utf-8")
            dev_changelog.write_text(
                "# Developer Changelog\n\n"
                "## Next Release\n\n"
                "<!-- insertion marker -->\n"
                "## [v1.0.0](release-url) - 2025-01-01\n",
                encoding="utf-8",
            )
            (changes_dir / "developer").mkdir()
            developer_fragment = (
                changes_dir / "developer" / "ci.workflow.md"
            )
            developer_fragment.write_text(
                "- Enforce changelog audiences.\n", encoding="utf-8"
            )

            with (
                mock.patch.object(
                    CHANGELOG, "USER_CHANGELOG", user_changelog
                ),
                mock.patch.object(
                    CHANGELOG, "DEV_CHANGELOG", dev_changelog
                ),
                mock.patch.object(CHANGELOG, "CHANGES_DIR", changes_dir),
            ):
                CHANGELOG.check_release_changelogs()
                CHANGELOG.update_changelogs("v1.1.0")

            self.assertEqual(
                user_text, user_changelog.read_text(encoding="utf-8")
            )
            dev_text = dev_changelog.read_text(encoding="utf-8")
            self.assertIn("## [v1.1.0]", dev_text)
            self.assertIn("- Enforce changelog audiences.", dev_text)
            self.assertFalse(developer_fragment.exists())

    def test_prepared_release_requires_stamp_and_no_fragments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            user_changelog = root / "CHANGELOG.md"
            dev_changelog = root / "DEV-CHANGELOG.md"
            changes_dir = root / "changes"
            changes_dir.mkdir()

            user_changelog.write_text(
                "# Changelog\n\n## Next Release\n",
                encoding="utf-8",
            )
            dev_changelog.write_text(
                "# Developer Changelog\n\n"
                "## Next Release\n\n"
                "## [v1.1.0](release-url) - 2025-02-01\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    CHANGELOG, "USER_CHANGELOG", user_changelog
                ),
                mock.patch.object(
                    CHANGELOG, "DEV_CHANGELOG", dev_changelog
                ),
                mock.patch.object(CHANGELOG, "CHANGES_DIR", changes_dir),
            ):
                CHANGELOG.check_prepared_changelogs("v1.1.0")

                legacy_fragment = changes_dir / "legacy.internal.md"
                legacy_fragment.write_text(
                    "- Legacy fragment.\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    SystemExit, "unprocessed changelog fragments"
                ):
                    CHANGELOG.check_prepared_changelogs("v1.1.0")

                legacy_fragment.unlink()
                with self.assertRaisesRegex(SystemExit, "not prepared"):
                    CHANGELOG.check_prepared_changelogs("v1.2.0")


if __name__ == "__main__":
    unittest.main()
