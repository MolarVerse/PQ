"""scripts/perf_local.py: the parts that decide what is built, measured and pinned."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "perf_local.py"

spec = importlib.util.spec_from_file_location("perf_local", SCRIPT)
perf_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_local)

GIT_ENVIRONMENT = dict(
    os.environ,
    GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
    GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
    GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
)  # fmt: skip


def git(repo, *arguments):
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-C", str(repo), *arguments],
        check=True, capture_output=True, env=GIT_ENVIRONMENT,
    )  # fmt: skip


def commit_of(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, env=GIT_ENVIRONMENT,
    ).stdout.strip()  # fmt: skip


class BenchmarkNamesTest(unittest.TestCase):
    def test_the_target_name_lowers_the_first_letter(self):
        text = "set(perf_benchmarks\n    perfInterWater.cpp\n    perfAdam.cpp\n)\n"

        self.assertEqual(["interWater", "adam"], perf_local.benchmark_names(text))

    def test_other_text_of_the_file_is_not_read(self):
        text = (
            "set(perf_benchmarks\n    perfAdam.cpp\n)\n"
            "# perfNotListed.cpp\nadd_executable(perf_x perfOther.cpp)\n"
        )

        self.assertEqual(["adam"], perf_local.benchmark_names(text))

    def test_no_list_gives_no_names(self):
        self.assertEqual([], perf_local.benchmark_names("project(x)\n"))

    def test_every_benchmark_source_of_the_repository_is_listed(self):
        names = perf_local.benchmark_names(
            (ROOT / perf_local.PERF_CMAKE).read_text()
        )
        sources = {
            p.stem[len("perf"):]
            for p in (ROOT / "benchmarks" / "perf").glob("perf[A-Z]*.cpp")
        }
        listed = {n[0].upper() + n[1:] for n in names}

        # perfAlloc.hpp, perfBenchSetup.hpp and the like are headers, not .cpp
        self.assertEqual(sources, listed)
        self.assertEqual(len(names), len(set(names)))


class SelectTest(unittest.TestCase):
    NAMES = ["interWater", "interWaterN216", "outputWritersN100", "adam"]

    def test_no_pattern_selects_everything(self):
        self.assertEqual(self.NAMES, perf_local.select(self.NAMES, []))

    def test_a_pattern_is_a_case_insensitive_substring(self):
        self.assertEqual(
            ["interWater", "interWaterN216"],
            perf_local.select(self.NAMES, ["INTERWATER"]),
        )

    def test_several_patterns_are_alternatives_in_the_original_order(self):
        self.assertEqual(
            ["interWater", "interWaterN216", "adam"],
            perf_local.select(self.NAMES, ["adam", "water"]),
        )

    def test_nothing_matches(self):
        self.assertEqual([], perf_local.select(self.NAMES, ["zzz"]))


class CpuTest(unittest.TestCase):
    def test_cpu_lists(self):
        self.assertEqual([2, 18], perf_local.parse_cpu_list("2,18\n"))
        self.assertEqual([4, 5, 6, 7, 12], perf_local.parse_cpu_list("4-7,12"))
        self.assertEqual([3], perf_local.parse_cpu_list("3"))

    def test_proc_stat_busy_is_everything_but_idle_and_iowait(self):
        text = (
            "cpu  100 0 100 800 100 0 0 0 0 0\n"
            "cpu0 10 0 10 70 10 0 0 0 0 0\n"
            "cpu1 50 0 0 40 0 10 0 0 0 0\n"
            "intr 1 2 3\n"
        )

        self.assertEqual(
            {0: (20, 100), 1: (60, 100)}, perf_local.parse_proc_stat(text)
        )

    def test_busy_fraction_of_a_window(self):
        before = {0: (10, 100), 1: (0, 100)}
        after = {0: (60, 200), 1: (0, 200), 2: (5, 5)}

        self.assertEqual(
            {0: 0.5, 1: 0.0}, perf_local.busy_fractions(before, after)
        )

    def test_a_window_without_time_is_not_a_division_by_zero(self):
        self.assertEqual(
            {0: 0.0}, perf_local.busy_fractions({0: (1, 5)}, {0: (1, 5)})
        )

    def test_the_least_loaded_cpu_is_chosen(self):
        siblings = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
        busy = {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.9}

        self.assertEqual(2, perf_local.choose_cpu([0, 1, 2, 3], siblings, busy))

    def test_a_busy_hyper_thread_sibling_counts_against_a_cpu(self):
        siblings = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
        # cpu 3 is idle, but its sibling cpu 1 is busy; cpu 0 and 2 are a bit
        # busy themselves, but the core as a whole is the quieter one
        busy = {0: 0.1, 1: 0.8, 2: 0.0, 3: 0.0}

        self.assertEqual(2, perf_local.choose_cpu([0, 1, 2, 3], siblings, busy))

    def test_ties_go_to_the_highest_cpu(self):
        siblings = {c: [c] for c in range(4)}

        self.assertEqual(3, perf_local.choose_cpu([0, 1, 2, 3], siblings, {}))

    def test_only_allowed_cpus_are_considered(self):
        siblings = {c: [c] for c in range(4)}
        busy = {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5}

        self.assertEqual(1, perf_local.choose_cpu([1, 2], siblings, busy))


class WarningsTest(unittest.TestCase):
    def test_a_quiet_machine_has_no_warning(self):
        self.assertEqual(
            [], perf_local.environment_warnings(0.1, "performance", 0.0, 0.0)
        )
        self.assertEqual([], perf_local.environment_warnings(0.1, None, 0.0, 0.0))

    def test_each_condition_gives_its_own_warning(self):
        self.assertEqual(
            1, len(perf_local.environment_warnings(3.0, "performance", 0.0, 0.0))
        )
        self.assertEqual(
            1, len(perf_local.environment_warnings(0.0, "powersave", 0.0, 0.0))
        )
        self.assertEqual(
            1, len(perf_local.environment_warnings(0.0, "performance", 0.5, 0.0))
        )
        self.assertEqual(
            1, len(perf_local.environment_warnings(0.0, "performance", 0.0, 0.5))
        )
        self.assertEqual(
            3, len(perf_local.environment_warnings(3.0, "powersave", 0.5, 0.5))
        )


class GitTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.tmp = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def make_repository(self, name, files, submodules=()):
        repo = self.tmp / name
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        for file, text in files.items():
            (repo / file).write_text(text)
        for path, other in submodules:
            git(repo, "submodule", "add", "-q", str(other), path)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
        return repo

    def test_an_explicit_ref_is_resolved_to_its_commit(self):
        repo = self.make_repository("r", {"a": "1"})

        self.assertEqual(commit_of(repo), perf_local.resolve_base(repo, "HEAD"))

    def test_the_default_base_is_the_merge_base_with_dev(self):
        repo = self.make_repository("r", {"a": "1"})
        git(repo, "branch", "dev")
        base = commit_of(repo)
        (repo / "a").write_text("2")
        git(repo, "commit", "-q", "-am", "change")

        self.assertEqual(base, perf_local.resolve_base(repo, None))

    def test_no_dev_branch_and_no_ref_is_an_error(self):
        repo = self.make_repository("r", {"a": "1"})

        with self.assertRaises(perf_local.Failure):
            perf_local.resolve_base(repo, None)

    def test_a_wrong_ref_is_an_error(self):
        repo = self.make_repository("r", {"a": "1"})

        with self.assertRaises(perf_local.Failure):
            perf_local.resolve_base(repo, "no-such-ref")

    def test_the_export_contains_the_files_of_the_commit_not_the_working_tree(self):
        repo = self.make_repository("r", {"a": "committed"})
        (repo / "a").write_text("edited, not committed")
        (repo / "untracked").write_text("x")
        destination = self.tmp / "out"

        perf_local.export_tree(repo, commit_of(repo), destination)

        self.assertEqual("committed", (destination / "a").read_text())
        self.assertFalse((destination / "untracked").exists())
        self.assertFalse((destination / ".git").exists())

    def test_the_export_contains_nested_submodules_at_the_pinned_commit(self):
        leaf = self.make_repository("leaf", {"leaf.txt": "leaf v1"})
        middle = self.make_repository(
            "middle", {"middle.txt": "m"}, submodules=[("ext/leaf", leaf)]
        )
        top = self.make_repository(
            "top", {"top.txt": "t"}, submodules=[("external/middle", middle)]
        )
        git(top / "external" / "middle", "submodule", "update", "--init")

        # the checkout moves on, the export has to use the commit that is pinned
        git(top / "external" / "middle" / "ext" / "leaf", "commit", "-q",
            "--allow-empty", "-m", "later")  # fmt: skip
        destination = self.tmp / "out"

        perf_local.export_tree(top, commit_of(top), destination)

        self.assertEqual("t", (destination / "top.txt").read_text())
        self.assertEqual("m", (destination / "external/middle/middle.txt").read_text())
        self.assertEqual(
            "leaf v1",
            (destination / "external/middle/ext/leaf/leaf.txt").read_text(),
        )
        self.assertEqual([], list(destination.rglob(".git")))

    def test_a_submodule_that_is_not_checked_out_is_a_clear_error(self):
        leaf = self.make_repository("leaf", {"f": "1"})
        top = self.make_repository("top", {"t": "1"}, submodules=[("ext", leaf)])
        # the directory is there, but it is not a checkout any more
        (top / "ext" / ".git").unlink()

        with self.assertRaises(perf_local.Failure) as caught:
            perf_local.export_tree(top, commit_of(top), self.tmp / "out")

        self.assertIn("submodule update", str(caught.exception))


class StageTest(unittest.TestCase):
    def test_only_the_chosen_binaries_that_exist_are_linked_with_the_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            perf = build / "benchmarks" / "perf"
            perf.mkdir(parents=True)
            for name in ("perf_a", "perf_b", "libperf_timer.so"):
                (perf / name).write_text("x")
            staged = Path(directory) / "stage"
            staged.mkdir()
            (staged / "old").write_text("left over")

            linked = perf_local.stage(staged, build, ["a", "c"])

            self.assertEqual(["a"], linked)
            self.assertEqual(
                ["libperf_timer.so", "perf_a"],
                sorted(p.name for p in staged.iterdir()),
            )
            self.assertTrue((staged / "perf_a").is_symlink())


class EigenTest(unittest.TestCase):
    def test_eigen_is_reused_only_for_the_same_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tree, text in (("a", "v1"), ("b", "v1"), ("c", "v2")):
                (root / tree / ".cmake").mkdir(parents=True)
                (root / tree / ".cmake" / "eigen.cmake").write_text(text)
            (root / "build" / "_deps" / "eigen-src").mkdir(parents=True)

            self.assertEqual(
                root / "build" / "_deps" / "eigen-src",
                perf_local.reusable_eigen(root / "a", root / "build", root / "b"),
            )
            self.assertIsNone(
                perf_local.reusable_eigen(root / "a", root / "build", root / "c")
            )
            self.assertIsNone(
                perf_local.reusable_eigen(root / "a", root / "nowhere", root / "b")
            )


if __name__ == "__main__":
    unittest.main()
