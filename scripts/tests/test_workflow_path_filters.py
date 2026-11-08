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


class ReleasePullRequestTests(unittest.TestCase):
    """A release pull request holds every change since the last release: the
    checks that lint or compare the whole diff do not run on it (#1028)."""

    @staticmethod
    def job_condition(workflow, job):
        text = (WORKFLOWS / workflow).read_text()
        match = re.search(rf"^  {job}:\n(?:.*\n)*?    if: >-\n((?:      .*\n)+)", text, re.MULTILINE)
        assert match, f"{workflow}: the {job} job has no multi-line if"
        return " ".join(match.group(1).split())

    def test_devops_cpp_check_is_skipped_for_release_pull_requests(self):
        condition = self.job_condition("devops_cpp_check.yml", "cpp-check")
        self.assertIn("needs.changes.outputs.relevant == 'true'", condition)
        self.assertIn("!(github.event_name == 'pull_request' && startsWith(github.head_ref, 'release/'))", condition)

    def test_clang_build_is_skipped_for_release_pull_requests(self):
        condition = self.job_condition("clang_build.yml", "clang-build")
        self.assertIn("!(github.event_name == 'pull_request' && startsWith(github.head_ref, 'release/'))", condition)

    def test_the_performance_gate_is_skipped_for_release_and_bot_pull_requests(self):
        condition = self.job_condition("perf.yml", "perf-regression")
        self.assertIn("github.event_name != 'pull_request'", condition)
        self.assertIn("startsWith(github.event.pull_request.head.ref, 'pq-bot/')", condition)
        self.assertIn("startsWith(github.event.pull_request.head.ref, 'release/')", condition)

    def test_the_linked_issue_check_is_skipped_for_release_pull_requests(self):
        # the release pull request has the body "Release vX.Y.Z." and no issue
        condition = self.job_condition("gh_checks.yml", "check-linked-issue")
        self.assertIn("github.event_name != 'pull_request'", condition)
        self.assertIn("!startsWith(github.head_ref, 'release/')", condition)

    def test_the_gate_jobs_still_pass_when_their_job_is_skipped(self):
        # the gates only fail on "failure", so a skipped job is a pass
        for workflow, result in (("devops_cpp_check.yml", "needs.cpp-check.result"), ("lint.yml", "needs.lint.result")):
            self.assertIn(f'if [[ "${{{{ {result} }}}}" == "failure" ]]', (WORKFLOWS / workflow).read_text())


class ChangelogWorkflowTests(unittest.TestCase):
    """The label skip-changelog waives the fragment, not the script tests (#1016)."""

    def setUp(self):
        self.text = (WORKFLOWS / "changelog.yml").read_text()

    def test_adding_or_removing_the_label_re_runs_the_check(self):
        self.assertIn("types: [opened, reopened, synchronize, labeled, unlabeled]", self.text)

    def test_only_the_skip_label_re_runs_the_job_on_a_label_event(self):
        self.assertIn("github.event.label.name == 'skip-changelog'", self.text)

    def test_the_label_skips_the_fragment_step_only(self):
        step = self.text.split("- name: Check changelog fragment", 1)[1].split("- name:", 1)[0]
        self.assertIn("!contains(github.event.pull_request.labels.*.name, 'skip-changelog')", step)
        tests = self.text.split("- name: Test changelog tooling", 1)[1].split("- name:", 1)[0]
        self.assertNotIn("\n        if:", tests)


if __name__ == "__main__":
    unittest.main()
