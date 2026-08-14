#!/usr/bin/env python3
"""Tests for release reviewer extraction."""

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from release_reviewers import extract_reviewers


class ReleaseReviewersTest(unittest.TestCase):
    def test_extracts_unique_human_authors(self):
        notes = """## What's Changed
* First change by @ape33 in https://github.com/MolarVerse/PQ/pull/1
* Second change by @97gamjak in https://github.com/MolarVerse/PQ/pull/2
* Follow-up by @ape33 in https://github.com/MolarVerse/PQ/pull/3
* Release by @pq-release-bot[bot] in https://github.com/MolarVerse/PQ/pull/4
"""

        self.assertEqual(
            ["ape33", "97gamjak"],
            extract_reviewers(notes, "MolarVerse/PQ"),
        )

    def test_ignores_other_repositories_and_unstructured_mentions(self):
        notes = """## What's Changed
* Other repository by @outside in https://github.com/Other/PQ/pull/5
Mention @maintainer outside a generated entry.
"""

        self.assertEqual([], extract_reviewers(notes, "MolarVerse/PQ"))

    def test_uses_the_generated_entry_author(self):
        notes = (
            "* Title by @incorrect in https://github.com/MolarVerse/PQ/pull/6 "
            "by @actual in https://github.com/MolarVerse/PQ/pull/7\n"
        )

        self.assertEqual(
            ["actual"],
            extract_reviewers(notes, "MolarVerse/PQ"),
        )


if __name__ == "__main__":
    unittest.main()
