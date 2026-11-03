import re
import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
NEGATED_PATTERN = re.compile(r"^\s*-\s*['\"]!")


class PathFilterTests(unittest.TestCase):
    def test_no_dorny_filter_uses_a_negated_pattern(self):
        # dorny/paths-filter ORs the patterns of a filter, and a negated pattern
        # on its own matches every changed file outside the excluded path, so
        # `- '!external/**'` made "relevant" true for any change at all.
        checked = 0
        for workflow in sorted(WORKFLOWS.glob("*.yml")):
            text = workflow.read_text()
            if "dorny/paths-filter" not in text:
                continue
            checked += 1
            for number, line in enumerate(text.splitlines(), 1):
                self.assertIsNone(NEGATED_PATTERN.match(line), f"{workflow.name}:{number}: {line.strip()}")
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
