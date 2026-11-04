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


class LintWorkflowTests(unittest.TestCase):
    def lint_condition(self):
        text = (WORKFLOWS / "lint.yml").read_text()
        match = re.search(r"^  lint:\n(?:.*\n)*?    if: >-\n((?:      .*\n)+)", text, re.MULTILINE)
        self.assertIsNotNone(match, "the lint job has no multi-line if")
        return " ".join(match.group(1).split())

    def test_lint_is_skipped_for_release_pull_requests_and_pushes_to_main(self):
        # such a run lints every file of the release (812 for 1.0.0), serially,
        # and was killed after half an hour
        condition = self.lint_condition()
        self.assertIn("needs.changes.outputs.relevant == 'true'", condition)
        self.assertIn("!(github.event_name == 'push' && github.ref == 'refs/heads/main')", condition)
        self.assertIn("!(github.event_name == 'pull_request' && startsWith(github.head_ref, 'release/'))", condition)

    def test_the_lint_gate_still_passes_when_lint_is_skipped(self):
        text = (WORKFLOWS / "lint.yml").read_text()
        self.assertIn('if [[ "${{ needs.lint.result }}" == "failure" ]]', text)


if __name__ == "__main__":
    unittest.main()
