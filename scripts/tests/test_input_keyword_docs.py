"""The input keywords the parsers register and the ones the user guide documents agree.

The registered names are read from the parser sources (``.name = "..."`` of
the key metadata and keys registered by name), the documented ones from the
``Key`` boxes of ``inputFile.rst``. A keyword that is documented but not
registered is rejected by PQ as invalid; one that is registered but not
documented cannot be found by users. Dashes and underscores are the same
(the reader normalises them).
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARSERS = ROOT / "src" / "input"
GUIDE = ROOT / "docs" / "sphinx" / "src" / "userGuide" / "inputFile.rst"

# registered on purpose without documentation
UNDOCUMENTED = {
    # accepted but never read, see the issue about it; documenting it would
    # promise an effect it does not have
    "use_force_conv",
    # the registration function exists but is not called (keyword removed for
    # now, see issues 999 and 1012)
    "qm_force_distribution",
    # accepts only 3 / 3d (every other value is an input error), so there is
    # nothing for a user to choose
    "dim",
}


def normalise(keyword):
    return keyword.strip().lower().replace("-", "_")


def registered_keywords():
    names = set()

    for path in PARSERS.rglob("*.cpp"):
        text = path.read_text()
        names.update(
            normalise(name)
            for name in re.findall(r'\.name\s*=\s*"([A-Za-z0-9_\-]+)"', text)
        )
        names.update(
            normalise(name)
            for name in re.findall(r'std::string\("([A-Za-z0-9_\-]+)"\)', text)
        )

    return names


def documented_keywords():
    lines = GUIDE.read_text().splitlines()
    names = set()

    for index, line in enumerate(lines):
        if not line.strip().startswith(".. admonition:: Key"):
            continue

        for following in lines[index + 1 : index + 8]:
            match = re.match(r"\s+([A-Za-z0-9_\-]+)\*?\s*=", following)
            if match:
                names.add(normalise(match.group(1)))
                break

    return names


class InputKeywordDocsTest(unittest.TestCase):
    def setUp(self):
        self.registered = registered_keywords()
        self.documented = documented_keywords()

    def test_the_sources_are_found(self):
        self.assertGreater(len(self.registered), 100)
        self.assertGreater(len(self.documented), 100)

    def test_every_documented_keyword_is_accepted_by_the_parser(self):
        self.assertEqual(
            [],
            sorted(self.documented - self.registered),
            "documented in inputFile.rst but not registered by any parser",
        )

    def test_every_registered_keyword_is_documented(self):
        self.assertEqual(
            [],
            sorted(self.registered - self.documented - UNDOCUMENTED),
            "registered by a parser but not documented in inputFile.rst",
        )

    def test_the_allowed_exceptions_are_still_exceptions(self):
        self.assertEqual(
            set(),
            UNDOCUMENTED & self.documented,
            "documented now, remove it from UNDOCUMENTED",
        )
        self.assertEqual(
            set(),
            UNDOCUMENTED - self.registered,
            "no longer registered, remove it from UNDOCUMENTED",
        )


if __name__ == "__main__":
    unittest.main()
