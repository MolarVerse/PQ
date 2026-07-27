import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "update_changelog.py"
SPEC = importlib.util.spec_from_file_location("update_changelog", SCRIPT)
CHANGELOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHANGELOG)


class ChangelogTests(unittest.TestCase):
    def test_user_notes_require_a_bullet(self):
        self.assertFalse(
            CHANGELOG.has_user_release_notes(
                ["", "### Bug Fixes", "", "No release notes yet."]
            )
        )
        self.assertTrue(
            CHANGELOG.has_user_release_notes(
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
            CHANGELOG.has_user_release_notes(
                stamped[next_index + 1 : marker_index]
            )
        )

    def test_release_updates_both_changelogs_and_consumes_fragments(self):
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
                "- Stale preview.\n\n"
                "<!-- insertion marker -->\n"
                "## [v1.0.0](release-url) - 2025-01-01\n",
                encoding="utf-8",
            )
            fragment = changes_dir / "legacy.test.md"
            fragment.write_text("- Cover the parser.\n", encoding="utf-8")

            generated = {
                section: [] for section in CHANGELOG.ORDER
            }
            generated["Internal"] = ["- Refactor the parser."]

            with (
                mock.patch.object(
                    CHANGELOG, "USER_CHANGELOG", user_changelog
                ),
                mock.patch.object(
                    CHANGELOG, "DEV_CHANGELOG", dev_changelog
                ),
                mock.patch.object(CHANGELOG, "CHANGES_DIR", changes_dir),
                mock.patch.object(
                    CHANGELOG, "run_git_cliff", return_value=generated
                ),
            ):
                CHANGELOG.update_changelogs("v1.1.0")

            user_text = user_changelog.read_text(encoding="utf-8")
            dev_text = dev_changelog.read_text(encoding="utf-8")

            self.assertIn("## Next Release\n\n<!-- insertion marker -->", user_text)
            self.assertIn("- Fix trajectory output.", user_text)
            self.assertIn("- Refactor the parser.", dev_text)
            self.assertIn("- Cover the parser.", dev_text)
            self.assertNotIn("- Stale preview.", dev_text)
            self.assertFalse(fragment.exists())


if __name__ == "__main__":
    unittest.main()
