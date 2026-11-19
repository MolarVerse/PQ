"""scripts/perf_scenarios.py: scenario files, input overrides, counts and paired timing."""

import importlib.util
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT = ROOT / "scripts" / "perf_scenarios.py"
spec = importlib.util.spec_from_file_location("perf_scenarios", SCRIPT)
perf_scenarios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_scenarios)


def write_script(path, text):
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


ENTRY = {"name": "a", "source": "examples/x", "input": "run.in",
         "steps": 10, "count_steps": 1}  # fmt: skip


class LoadScenariosTest(unittest.TestCase):
    def test_a_scenario_is_read(self):
        scenarios = perf_scenarios.load_scenarios(json.dumps([ENTRY]))

        self.assertEqual("a", scenarios[0].name)
        self.assertEqual(10, scenarios[0].steps)
        self.assertEqual("", scenarios[0].description)

    def test_a_missing_field_is_an_error(self):
        entry = dict(ENTRY)
        del entry["count_steps"]

        with self.assertRaisesRegex(ValueError, "count_steps"):
            perf_scenarios.load_scenarios(json.dumps([entry]))

    def test_duplicate_names_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            perf_scenarios.load_scenarios(json.dumps([ENTRY, ENTRY]))

    def test_an_unknown_field_is_an_error(self):
        with self.assertRaises(TypeError):
            perf_scenarios.load_scenarios(json.dumps([dict(ENTRY, speed="fast")]))

    def test_the_scenarios_of_the_repository_prepare_from_their_sources(self):
        scenarios = perf_scenarios.load_scenarios(
            (ROOT / perf_scenarios.SCENARIOS_FILE).read_text()
        )
        self.assertGreater(len(scenarios), 0)

        with tempfile.TemporaryDirectory() as directory:
            for scenario in scenarios:
                template = perf_scenarios.prepare(
                    scenario, ROOT, Path(directory) / scenario.name, scenario.steps
                )
                text = (template / scenario.input).read_text()

                self.assertIn(f"nstep = {scenario.steps};", text, scenario.name)
                self.assertIn("random_seed = 1;", text, scenario.name)
                self.assertGreater(scenario.count_steps, 0, scenario.name)
                self.assertLessEqual(scenario.count_steps, scenario.steps, scenario.name)


class ApplyOverridesTest(unittest.TestCase):
    def test_a_present_key_is_changed_in_place(self):
        text = "jobtype = mm-md;\n  nstep = 25000; timestep = 0.5;\n"

        self.assertEqual(
            "jobtype = mm-md;\n  nstep = 10; timestep = 0.5;\n",
            perf_scenarios.apply_overrides(text, {"nstep": 10}),
        )

    def test_a_missing_key_is_appended(self):
        result = perf_scenarios.apply_overrides("a = 1;\n", {"random_seed": 1})

        self.assertEqual("a = 1;\nrandom_seed = 1;\n", result)

    def test_a_comment_is_not_changed_and_does_not_count_as_the_key(self):
        text = "# nstep = 5;\nnstep = 7; # was nstep = 9;\n"

        result = perf_scenarios.apply_overrides(text, {"nstep": 3})

        self.assertEqual("# nstep = 5;\nnstep = 3; # was nstep = 9;\n", result)

    def test_a_key_that_is_only_in_a_comment_is_appended(self):
        result = perf_scenarios.apply_overrides("# nstep = 5;\n", {"nstep": 3})

        self.assertEqual("# nstep = 5;\nnstep = 3;\n", result)

    def test_longer_and_similar_keys_are_left_alone(self):
        text = "nstep_x = 1; my-nstep = 2; nreset_angular = 3; nreset = 4;\n"

        result = perf_scenarios.apply_overrides(text, {"nstep": 9, "nreset": 0})

        self.assertEqual(
            "nstep_x = 1; my-nstep = 2; nreset_angular = 3; nreset = 0;\n"
            "nstep = 9;\n",
            result,
        )

    def test_spaces_around_the_equals_sign_are_kept(self):
        self.assertEqual(
            "nstep=10;\n", perf_scenarios.apply_overrides("nstep=99;", {"nstep": 10})
        )


class PrepareTest(unittest.TestCase):
    def test_a_directory_and_a_zip_member_give_the_same_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "examples" / "x").mkdir(parents=True)
            (root / "examples" / "x" / "run.in").write_text("nstep = 99;\n")
            (root / "examples" / "x" / "data.dat").write_text("d")
            with zipfile.ZipFile(root / "tests.zip", "w") as zipped:
                zipped.writestr("case/run.in", "nstep = 99;\n")
                zipped.writestr("case/data.dat", "d")
                zipped.writestr("case/ref_data/ref.en", "reference")
                zipped.writestr("other/run.in", "not this one\n")

            from_directory = perf_scenarios.prepare(
                perf_scenarios.Scenario("d", "examples/x", "run.in", 5, 1),
                root, root / "out1", 5)
            from_zip = perf_scenarios.prepare(
                perf_scenarios.Scenario("z", "tests.zip:case", "run.in", 5, 1),
                root, root / "out2", 5)  # fmt: skip

            for template in (from_directory, from_zip):
                self.assertEqual("d", (template / "data.dat").read_text())
                self.assertIn("nstep = 5;", (template / "run.in").read_text())
            self.assertFalse((from_zip / "ref_data").exists())
            self.assertFalse((root / "out2" / "other").exists())
            # the source is not touched
            self.assertEqual("nstep = 99;\n", (root / "examples/x/run.in").read_text())

    def test_a_missing_input_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "examples" / "x").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "run.in"):
                perf_scenarios.prepare(
                    perf_scenarios.Scenario("d", "examples/x", "run.in", 5, 1),
                    root, root / "out", 5)  # fmt: skip


class CountVerdictTest(unittest.TestCase):
    def test_a_rise_above_the_threshold_is_a_regression(self):
        verdict = perf_scenarios.count_verdict("a", 1000, 1020, 1.0)

        self.assertEqual("regression", verdict.status)
        self.assertAlmostEqual(2.0, verdict.percent)

    def test_a_rise_within_the_threshold_is_ok(self):
        self.assertEqual("ok", perf_scenarios.count_verdict("a", 1000, 1009, 1.0).status)

    def test_a_drop_beyond_the_threshold_is_an_improvement(self):
        self.assertEqual(
            "improvement", perf_scenarios.count_verdict("a", 1000, 900, 1.0).status
        )

    def test_a_failed_run_is_not_a_number(self):
        self.assertEqual("failed", perf_scenarios.count_verdict("a", 1000, None, 1.0).status)
        self.assertEqual(
            "base failed", perf_scenarios.count_verdict("a", None, 1000, 1.0).status
        )


class RunTest(unittest.TestCase):
    def test_a_run_gives_its_cpu_time_and_a_failure_gives_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template"
            template.mkdir()
            (template / "run.in").write_text("x")
            good = write_script(
                root / "good", "i=0; while [ $i -lt 20000 ]; do i=$((i+1)); done\n")
            bad = write_script(root / "bad", "exit 3\n")  # fmt: skip

            ns = perf_scenarios.run_cpu_ns(good, template, "run.in", None)

            self.assertGreater(ns, 1_000_000)
            self.assertIsNone(perf_scenarios.run_cpu_ns(bad, template, "run.in", None))

    def test_a_run_gets_its_own_copy_of_the_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template"
            template.mkdir()
            (template / "run.in").write_text("x")
            writer = write_script(root / "writer", "echo output > out.txt\n")

            perf_scenarios.run_cpu_ns(writer, template, "run.in", None)

            self.assertEqual(["run.in"], [p.name for p in template.iterdir()])

    def test_the_instructions_are_read_from_the_callgrind_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template"
            template.mkdir()
            valgrind = write_script(
                root / "valgrind", 'echo "==1== I   refs:      1,234,567" >&2\n')
            failing = write_script(root / "failing", "exit 1\n")  # fmt: skip

            self.assertEqual(
                1234567,
                perf_scenarios.instruction_count(str(valgrind), "pq", template, "run.in"),
            )
            self.assertIsNone(
                perf_scenarios.instruction_count(str(failing), "pq", template, "run.in")
            )


class TimeScenariosTest(unittest.TestCase):
    LOOP = "i=0; while [ $i -lt {n} ]; do i=$((i+1)); done\n"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.template = self.root / "template"
        self.template.mkdir()
        (self.template / "run.in").write_text("x")
        self.scenario = perf_scenarios.Scenario("s", "unused", "run.in", 1, 1)

    def tearDown(self):
        self.directory.cleanup()

    def time(self, base_text, pr_text, **options):
        base = write_script(self.root / "base", base_text)
        pr = write_script(self.root / "pr", pr_text)
        arguments = dict(rounds=3, copies=2, threshold=0.5, min_ms=1.0)
        arguments.update(options)

        return perf_scenarios.time_scenarios(
            [self.scenario], {"s": self.template}, pr, base, None,
            out=lambda _line: None, **arguments)  # fmt: skip

    def test_a_much_slower_pr_is_a_regression(self):
        verdicts, notes = self.time(
            self.LOOP.format(n=20000), self.LOOP.format(n=100000)
        )

        self.assertEqual([], notes)
        self.assertEqual("regression", verdicts[0].status)
        self.assertGreater(verdicts[0].ratio, 2.0)

    def test_the_same_program_on_both_sides_is_not_a_regression(self):
        text = self.LOOP.format(n=30000)

        verdicts, _ = self.time(text, text, threshold=0.5)

        self.assertNotEqual("regression", verdicts[0].status)
        self.assertAlmostEqual(1.0, verdicts[0].ratio, delta=0.5)

    def test_a_run_that_is_too_short_is_noted_not_judged(self):
        verdicts, notes = self.time("exit 0\n", "exit 0\n", min_ms=500.0)

        self.assertEqual([], verdicts)
        self.assertIn("too short", notes[0])

    def test_a_failing_side_is_noted_with_its_name(self):
        _, base_notes = self.time("exit 1\n", self.LOOP.format(n=1000))
        _, pr_notes = self.time(self.LOOP.format(n=1000), "exit 1\n")

        self.assertIn("the base", base_notes[0])
        self.assertIn("the PR", pr_notes[0])


class RenderTest(unittest.TestCase):
    def test_the_report_has_a_row_per_scenario_and_the_notes(self):
        counts = [
            perf_scenarios.count_verdict("a", 1000, 1050, 1.0),
            perf_scenarios.count_verdict("b", 1000, None, 1.0),
        ]

        report = perf_scenarios.render_report(counts, [], ["c: too short"])

        self.assertIn("| `a` | 1,000 | 1,050 | +5.00% | ❌ |", report)
        self.assertIn("| `b` | – | – | failed | ⚠️ |", report)
        self.assertIn("- c: too short", report)

    def test_a_count_line_names_a_regression(self):
        line = perf_scenarios.render_count(
            perf_scenarios.count_verdict("a", 1000, 1050, 1.0)
        )

        self.assertIn("+5.00%", line)
        self.assertIn("REGRESSION", line)


if __name__ == "__main__":
    unittest.main()
