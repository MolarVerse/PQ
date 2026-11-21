"""scripts/clang_tidy.sh: the files are checked in batches of fresh processes."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "clang_tidy.sh"

# records the arguments of every call, one line per call; FAKE_FAILING_CALLS is
# a space separated list of call numbers that exit with status 1
FAKE_CLANGD_TIDY = """#!/bin/sh
echo "$@" >> "$FAKE_CALLS"
call=$(wc -l < "$FAKE_CALLS")
for failing in $FAKE_FAILING_CALLS; do
    if [ "$failing" = "$call" ]; then
        echo "src/a.cpp:1:1: Warning: finding in call $call"
        exit 1
    fi
done
exit 0
"""


class ClangTidyBatchesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "clangd-tidy"
        fake.write_text(FAKE_CLANGD_TIDY)
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        self.calls = self.root / "calls.txt"

        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init", "-q")

    def tearDown(self):
        self.directory.cleanup()

    def _git(self, *arguments):
        subprocess.run(["git", "-C", str(self.repo), *arguments], check=True,
                       capture_output=True)

    def add_files(self, count):
        (self.repo / "src").mkdir(exist_ok=True)
        for index in range(count):
            (self.repo / "src" / f"f{index:02d}.cpp").write_text("int x;\n")
        self._git("add", "-A")

    def run_script(self, *arguments, failing=(), env=None):
        environment = dict(os.environ)
        environment.pop("CLANG_TIDY_BATCH_SIZE", None)
        environment.update(
            PATH=f"{self.bin}{os.pathsep}{environment['PATH']}",
            FAKE_CALLS=str(self.calls),
            FAKE_FAILING_CALLS=" ".join(str(call) for call in failing),
        )
        environment.update(env or {})
        self.calls.write_text("")

        return subprocess.run(
            ["bash", str(SCRIPT), "--all", "-p", "build", *arguments],
            cwd=self.repo, env=environment, capture_output=True, text=True,
            check=False)

    def recorded_calls(self):
        return [line.split() for line in self.calls.read_text().splitlines()]

    @staticmethod
    def files_of(call):
        return [item for item in call if item.endswith(".cpp")]

    def test_the_files_are_split_into_batches_that_cover_each_file_once(self):
        self.add_files(7)

        result = self.run_script("--batch-size", "3")

        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.recorded_calls()
        self.assertEqual([3, 3, 1], [len(self.files_of(c)) for c in calls])
        covered = [name for call in calls for name in self.files_of(call)]
        self.assertEqual(sorted(set(covered)), sorted(covered))
        self.assertEqual(7, len(covered))

    def test_every_batch_gets_the_same_flags(self):
        self.add_files(4)

        self.run_script("--batch-size", "2")

        for call in self.recorded_calls():
            self.assertIn("-p=build", call)
            self.assertIn("--fail-on-severity=hint", call)
            self.assertIn("--tqdm", call)
            self.assertIn("-j1", call)

    def test_each_batch_is_announced_in_the_log(self):
        self.add_files(5)

        result = self.run_script("--batch-size", "2")

        self.assertIn("Batches: 3 of up to 2 files", result.stdout)
        self.assertIn("Batch 1 of 3: files 1-2", result.stdout)
        self.assertIn("Batch 3 of 3: files 5-5", result.stdout)

    def test_zero_means_one_process_for_everything(self):
        self.add_files(5)

        result = self.run_script("--batch-size", "0")

        self.assertEqual(0, result.returncode)
        self.assertEqual([5], [len(self.files_of(c)) for c in self.recorded_calls()])
        self.assertNotIn("Batches:", result.stdout)

    def test_a_batch_size_of_at_least_the_number_of_files_is_one_process(self):
        self.add_files(5)

        self.run_script("--batch-size", "5")

        self.assertEqual(1, len(self.recorded_calls()))

    def test_the_default_is_30_files_per_batch(self):
        self.add_files(31)

        self.run_script()

        self.assertEqual([30, 1], [len(self.files_of(c)) for c in self.recorded_calls()])

    def test_the_environment_variable_sets_the_default(self):
        self.add_files(6)

        self.run_script(env={"CLANG_TIDY_BATCH_SIZE": "2"})

        self.assertEqual(3, len(self.recorded_calls()))

    def test_the_option_wins_over_the_environment_variable(self):
        self.add_files(6)

        self.run_script("--batch-size", "3", env={"CLANG_TIDY_BATCH_SIZE": "2"})

        self.assertEqual(2, len(self.recorded_calls()))

    def test_a_failing_batch_does_not_stop_the_others_but_fails_the_run(self):
        self.add_files(7)

        result = self.run_script("--batch-size", "3", failing=(2,))

        self.assertEqual(3, len(self.recorded_calls()))
        self.assertEqual(1, result.returncode)
        self.assertIn("finding in call 2", result.stdout)

    def test_all_batches_passing_exits_zero(self):
        self.add_files(7)

        self.assertEqual(0, self.run_script("--batch-size", "3").returncode)

    def test_a_single_process_run_passes_its_failure_on(self):
        self.add_files(3)

        result = self.run_script("--batch-size", "0", failing=(1,))

        self.assertEqual(1, result.returncode)

    def test_a_batch_size_that_is_not_a_number_is_refused(self):
        self.add_files(3)

        result = self.run_script("--batch-size", "many")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("--batch-size needs a number", result.stdout)
        self.assertEqual([], self.recorded_calls())

    def test_nothing_to_check_runs_nothing(self):
        result = self.run_script("--batch-size", "3")

        self.assertEqual(0, result.returncode)
        self.assertIn("No files to check", result.stdout)
        self.assertEqual([], self.recorded_calls())

    def test_the_log_file_has_the_batch_lines(self):
        self.add_files(4)

        self.run_script("--batch-size", "2")

        log = (self.repo / "clangd-tidy-report.log").read_text()
        self.assertIn("Batch 2 of 2", log)


if __name__ == "__main__":
    unittest.main()
