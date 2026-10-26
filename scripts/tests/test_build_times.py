import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

TOOL_DIR = Path(__file__).resolve().parents[1] / "pq_build_times"
sys.path.insert(0, str(TOOL_DIR))

import bt_compare as compare  # noqa: E402
import bt_detail as detail  # noqa: E402
import bt_fingerprint as fp  # noqa: E402
import bt_ninja as ninja  # noqa: E402
import bt_report as report  # noqa: E402
import bt_trace as trace  # noqa: E402
import bt_trace_report as trace_report  # noqa: E402
import bt_snapshot as snap  # noqa: E402
import bt_store as store  # noqa: E402
import pqbt  # noqa: E402


def run_stub(outputs):
    return lambda command, cwd=None: outputs.get(command[0], "")


def fingerprint(**changes):
    base = {
        "arch": "x86_64", "cpu_model": "Test CPU", "logical_cores": 8, "compiler": "g++ 15", "build_type": "Release",
        "target": "all", "cmake_args": ["-DA=1"], "jobs": 8, "ccache": False, "kernel": "7.0", "ram_gb": 32.0, "os": "Linux",
    }
    base.update(changes)
    return base


def make_snapshot(ident, wall=10.0, note="", fid="abc123abc123", commit="1234567890ab", dirty=False, steps=5, pairs=100, targets=None):
    scenarios = {name: {"wall_s": wall, "steps": steps, "cpu_s": wall * 4} for name, _ in report.WALL_PANELS}
    return {
        "schema_version": store.SCHEMA_VERSION, "kind": store.KIND, "id": ident,
        "created_at": f"{ident[:4]}-{ident[4:6]}-{ident[6:8]}T{ident[9:11]}:{ident[11:13]}:{ident[13:15]}Z",
        "note": note, "fingerprint_id": fid, "fingerprint": fingerprint(),
        "git": {"commit": commit, "dirty": dirty, "branch": "dev", "submodules": {}},
        "load": {"start": 0.1, "end": 0.2}, "target": "all", "repeat": 3,
        "targets": targets or {"leaf": "src/a.cpp", "header_top": "include/t.hpp", "header_median": "include/m.hpp"},
        "scenarios": scenarios,
        "include_graph": {"include_pairs": pairs, "project_pairs": pairs // 2},
    }


class FingerprintTests(unittest.TestCase):
    def test_the_id_is_stable_and_follows_the_keys_that_matter(self):
        base = fp.fingerprint_id(fingerprint())
        self.assertEqual(base, fp.fingerprint_id(fingerprint()))
        for key, value in (
            ("arch", "aarch64"), ("cpu_model", "Other"), ("logical_cores", 16), ("compiler", "clang 20"),
            ("build_type", "Debug"), ("target", "testX"), ("cmake_args", ["-DA=2"]), ("jobs", 4), ("ccache", True),
        ):
            self.assertNotEqual(base, fp.fingerprint_id(fingerprint(**{key: value})), key)

    def test_details_that_only_help_reading_do_not_change_the_id(self):
        base = fp.fingerprint_id(fingerprint())
        self.assertEqual(base, fp.fingerprint_id(fingerprint(kernel="9.9", ram_gb=1.0, os="Other OS")))

    def test_machine_specific_cache_paths_are_not_part_of_the_arguments(self):
        args = ["-DX=1", "-DFETCHCONTENT_BASE_DIR=/home/me/deps", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON", "-DA=2"]
        self.assertEqual(["-DA=2", "-DX=1"], fp.comparable_cmake_args(args))

    def test_the_cpu_model_comes_from_cpuinfo_or_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpuinfo"
            path.write_text("processor\t: 0\nmodel name\t: AMD  EPYC   7763\n")
            self.assertEqual("AMD EPYC 7763", fp.cpu_model(cpuinfo=str(path)))
            path.write_text("processor\t: 0\nCPU part\t: 0xd0c\n")
            run = run_stub({"lscpu": "Architecture: aarch64\nModel name:   Neoverse-N1\n"})
            self.assertEqual("Neoverse-N1", fp.cpu_model(run, cpuinfo=str(path)))

    def test_collect_records_the_toolchain(self):
        run = run_stub({"g++": "g++ (Ubuntu) 15.2.0\nCopyright", "cmake": "cmake version 4.2.3\n", "ninja": "1.13.2\n"})
        result = fp.collect("g++", "Release", "all", ["-DA=1", "-DFETCHCONTENT_BASE_DIR=/x"], 8, False, run)
        self.assertEqual("g++ (Ubuntu) 15.2.0", result["compiler"])
        self.assertEqual("cmake version 4.2.3", result["cmake"])
        self.assertEqual(["-DA=1"], result["cmake_args"])
        self.assertEqual(8, result["jobs"])
        self.assertIn("arch", result)

    def test_the_description_names_the_architecture_and_cpu(self):
        text = fp.describe(fingerprint())
        for part in ("x86_64", "Test CPU", "8 threads", "g++ 15", "Release", "target all", "-j8"):
            self.assertIn(part, text)


LOG = (
    "# ninja log v7\n"
    "0\t1000\t1\tsrc/a.o\th1\n"
    "0\t3000\t1\tsrc/b.o\th2\n"
    "3000\t4500\t1\tlib/liba.so\th3\n"
    "3000\t4500\t1\tlib/liba.so.1\th3\n"
)


class NinjaTests(unittest.TestCase):
    def test_log_lines_ignore_the_header_and_parse_into_entries(self):
        lines = ninja.log_lines(LOG)
        self.assertEqual(4, len(lines))
        entries = ninja.parse_log(lines)
        self.assertEqual({"src/a.o", "src/b.o", "lib/liba.so", "lib/liba.so.1"}, {e.output for e in entries})
        self.assertEqual([], ninja.parse_log(["garbage", "1\t2\tx\ty\tz"]))

    def test_steps_cpu_link_and_tail(self):
        result = ninja.summarise_steps(ninja.parse_log(ninja.log_lines(LOG)))
        self.assertEqual(3, result["steps"])   # the two outputs of one command count once
        self.assertEqual(5.5, result["cpu_s"])   # 1 + 3 + 1.5
        self.assertEqual(1.5, result["link_s"])
        self.assertEqual(1.5, result["tail_s"])   # from the last compile (3.0 s) to the end (4.5 s)

    def test_no_steps_give_zeros(self):
        self.assertEqual({"steps": 0, "cpu_s": 0, "link_s": 0, "tail_s": 0}, ninja.summarise_steps([]))

    def test_a_build_only_adds_new_lines(self):
        before = ninja.log_lines(LOG)
        after = ninja.log_lines(LOG + "5000\t5200\t2\tsrc/a.o\th9\n")
        self.assertEqual({"5000\t5200\t2\tsrc/a.o\th9"}, after - before)

    def test_deps_are_parsed_per_target(self):
        text = (
            "src/a.o: #deps 2, deps mtime 100 (VALID)\n    ../src/a.cpp\n    /usr/include/stdio.h\n\n"
            "lib/liba.so: #deps 0\n\n"
            "src/b.o: #deps 1, deps mtime 100 (STALE)\n    ../src/b.cpp\n\n"
        )
        self.assertEqual({"src/a.o": ["../src/a.cpp", "/usr/include/stdio.h"], "src/b.o": ["../src/b.cpp"]}, ninja.parse_deps(text))


class IncludeGraphTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self.build = os.path.join(self.root, "build")
        os.makedirs(self.build)
        self.deps = {
            "src/a.o": ["../src/a.cpp", "../include/core.hpp", "../include/only_a.hpp", "/usr/include/stdio.h",
                        "../external/mstd/x.hpp", "gen/generated.hpp", f"{self.root}/build/abs_gen.hpp"],
            "src/b.o": [f"{self.root}/src/b.cpp", "../include/core.hpp"],
            "src/c.o": ["../src/c.cpp", "../include/core.hpp", "../include/mid.hpp"],
            "tests/t.o": ["../tests/t.cpp", "../include/core.hpp", "../include/mid.hpp"],
            "lib/liba.so": ["ignored"],
        }

    def tearDown(self):
        self._tmp.cleanup()

    def graph(self, deps=None):
        return ninja.include_graph(deps or self.deps, self.root, self.build)

    def test_project_files_exclude_system_external_and_generated_files(self):
        fan_in = self.graph()["fan_in"]
        self.assertEqual(
            {"src/a.cpp", "src/b.cpp", "src/c.cpp", "tests/t.cpp", "include/core.hpp", "include/only_a.hpp", "include/mid.hpp"},
            set(fan_in),
        )
        self.assertEqual(4, fan_in["include/core.hpp"])
        self.assertEqual(2, fan_in["include/mid.hpp"])

    def test_metrics_count_objects_files_and_pairs(self):
        metrics = self.graph()["metrics"]
        self.assertEqual(4, metrics["objects"])
        self.assertEqual(7 + 2 + 3 + 3, metrics["include_pairs"])
        self.assertEqual(11, metrics["project_pairs"])
        self.assertEqual(7, metrics["project_files"])
        self.assertEqual(64, len(metrics["digest"]))

    def test_the_digest_follows_the_graph_not_the_checkout_location(self):
        first = self.graph()["metrics"]["digest"]
        self.assertEqual(first, self.graph()["metrics"]["digest"])
        changed = dict(self.deps, **{"src/b.o": self.deps["src/b.o"] + ["../include/mid.hpp"]})
        self.assertNotEqual(first, self.graph(changed)["metrics"]["digest"])

    def test_targets_are_deterministic(self):
        targets = ninja.choose_targets(self.graph())
        self.assertEqual("include/core.hpp", targets["header_top"])
        self.assertEqual("include/mid.hpp", targets["header_median"])   # lower middle of {mid (2), core (4)}
        self.assertTrue(targets["leaf"].startswith("src/"))
        self.assertEqual(targets, ninja.choose_targets(self.graph()))

    def test_ties_break_by_path_and_missing_candidates_give_none(self):
        deps = {"src/a.o": ["../src/a.cpp", "../include/z.hpp", "../include/a.hpp"], "src/b.o": ["../src/b.cpp", "../include/z.hpp", "../include/a.hpp"]}
        self.assertEqual("include/a.hpp", ninja.choose_targets(self.graph(deps))["header_top"])
        none = ninja.choose_targets(self.graph({"src/a.o": ["../src/a.cpp"]}))
        self.assertEqual((None, None), (none["header_top"], none["header_median"]))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_data_directory_precedence(self):
        self.assertEqual(Path("/cli"), store.data_dir("/cli", {store.ENV_VAR: "/env"}))
        self.assertEqual(Path("/env"), store.data_dir(None, {store.ENV_VAR: "/env"}))
        self.assertEqual(Path("/xdg/pq-build-times"), store.data_dir(None, {"XDG_DATA_HOME": "/xdg"}))
        self.assertTrue(str(store.data_dir(None, {})).endswith(".local/share/pq-build-times"))

    def test_snapshots_round_trip_in_order(self):
        for ident in ("20261003T100000Z", "20261001T100000Z", "20261002T100000Z"):
            store.save_snapshot(self.root, make_snapshot(ident))
        self.assertEqual(["20261001T100000Z", "20261002T100000Z", "20261003T100000Z"], [s["id"] for s in store.load_snapshots(self.root)])

    def test_unreadable_and_foreign_files_are_skipped(self):
        store.save_snapshot(self.root, make_snapshot("20261001T100000Z"))
        (self.root / "snapshots" / "broken.json").write_text("{not json")
        (self.root / "snapshots" / "foreign.json").write_text('{"kind": "something-else"}')
        (self.root / "snapshots" / "old.json").write_text('{"kind": "local-build-snapshot", "schema_version": 99}')
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            loaded = store.load_snapshots(self.root)
        self.assertEqual(1, len(loaded))
        self.assertEqual(3, errors.getvalue().count("skipping"))

    def test_series_are_separated_by_fingerprint(self):
        store.save_snapshot(self.root, make_snapshot("20261001T100000Z", fid="aaaaaaaaaaaa"))
        store.save_snapshot(self.root, make_snapshot("20261002T100000Z", fid="bbbbbbbbbbbb"))
        self.assertEqual(["20261001T100000Z"], [s["id"] for s in store.load_snapshots(self.root, "aaaaaaaaaaaa")])
        self.assertEqual({"aaaaaaaaaaaa", "bbbbbbbbbbbb"}, set(store.fingerprints(self.root)))

    def test_the_baseline_is_explicit_and_per_fingerprint(self):
        store.save_snapshot(self.root, make_snapshot("20261001T100000Z"))
        store.save_snapshot(self.root, make_snapshot("20261002T100000Z"))
        self.assertIsNone(store.baseline_snapshot(self.root, "abc123abc123"))
        store.set_baseline(self.root, "abc123abc123", "20261001T100000Z")
        self.assertEqual("20261001T100000Z", store.baseline_snapshot(self.root, "abc123abc123")["id"])
        self.assertIsNone(store.baseline_snapshot(self.root, "other"))

    def test_touch_targets_are_pinned_to_the_baseline_else_the_first_snapshot(self):
        first = make_snapshot("20261001T100000Z", targets={"leaf": "src/first.cpp"})
        second = make_snapshot("20261002T100000Z", targets={"leaf": "src/second.cpp"})
        self.assertEqual({}, store.pinned_targets(self.root, "abc123abc123"))
        store.save_snapshot(self.root, first)
        store.save_snapshot(self.root, second)
        self.assertEqual("src/first.cpp", store.pinned_targets(self.root, "abc123abc123")["leaf"])
        store.set_baseline(self.root, "abc123abc123", "20261002T100000Z")
        self.assertEqual("src/second.cpp", store.pinned_targets(self.root, "abc123abc123")["leaf"])


def entry(start_ms, end_ms, output, digest="h"):
    return ninja.LogEntry(start_ms, end_ms, 1, output, digest + output)


OBJ_A = "src/CMakeFiles/lib.dir/a.cpp.o"
OBJ_B = "src/CMakeFiles/lib.dir/b.cpp.o"
COLD_ENTRIES = [
    entry(0, 2000, "src/CMakeFiles/pq_pch.dir/cmake_pch.hxx.gch"),
    entry(2000, 6000, OBJ_A),
    entry(2000, 5000, OBJ_B),
    entry(6000, 6500, "src/liblib.so"),
    entry(6500, 7000, "apps/PQ"),
]


class FakeBuilder:
    source_root = "/src"
    build_dir = "/src/build-times"
    build_type = "Release"
    cmake_args = ["-DA=1"]
    jobs = 8
    target = "all"

    def __init__(self, deps_text=None, built=True):
        self.calls = []
        self.deps = deps_text or (
            f"{OBJ_A}: #deps 3, deps mtime 1 (VALID)\n    ../src/a.cpp\n    ../include/core.hpp\n    ../include/mid.hpp\n\n"
            f"{OBJ_B}: #deps 3, deps mtime 1 (VALID)\n    ../src/b.cpp\n    ../include/core.hpp\n    ../include/mid.hpp\n\n"
        )
        self.built = built
        self.clock = 0
        self.last_steps = []
        self.cold_entries = []
        self._touched = None
        self._touch_runs = {}
        self.on_touch = None

    def _result(self, steps, wall):
        return {"wall_s": wall, "steps": steps, "cpu_s": wall * 4, "link_s": 1.0, "tail_s": 0.5}

    def cold(self):
        self.calls.append("cold")
        result = self._result(100, 100.0)
        result["configure_s"] = 5.0
        self.last_steps = self.cold_entries = list(COLD_ENTRIES)
        return result

    def all_entries(self):
        return list(COLD_ENTRIES)

    def ensure_built(self):
        self.calls.append("ensure_built")
        if not self.built:
            raise snap.SnapshotError("nothing to measure yet")

    def build(self):
        self.calls.append("build")
        if self._touched:
            self._touch_runs[self._touched] = self._touch_runs.get(self._touched, 0) + 1
            first = self._touch_runs[self._touched] == 1
            self.last_steps = [entry(0, 1500, OBJ_A), entry(1500, 1800, "src/liblib.so")] if first else [entry(0, 9000, OBJ_B)]
        else:
            self.last_steps = []
        self._touched = None
        return self._result(3, 2.0)

    def deps_text(self):
        return self.deps

    def touch(self, relative):
        self.calls.append(f"touch {relative}")
        self._touched = relative
        if self.on_touch:
            self.on_touch()

    def compiler_path(self):
        return "g++"


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(fp, "default_run", lambda command, cwd=None: "")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []

    def tearDown(self):
        self._tmp.cleanup()

    def take(self, builder=None, scenarios=snap.SCENARIOS, repeat=2, overrides=None, loadavg=lambda: (0.1, 0.1, 0.1)):
        snapshot, self.detail = snap.take_snapshot(
            builder or FakeBuilder(), self.root, list(scenarios), repeat, 1, "a note", overrides or {}, False,
            now=lambda: datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc), loadavg=loadavg, log=self.log.append,
        )
        return snapshot

    def test_all_scenarios_run_in_order_and_touch_the_chosen_files(self):
        builder = FakeBuilder()
        result = self.take(builder)
        touch = lambda name: [f"touch {name}", "build", f"touch {name}", "build"]
        self.assertEqual(
            ["cold", "build", "build"] + touch("src/a.cpp") + touch("include/core.hpp") + touch("include/core.hpp"),
            builder.calls,
        )
        self.assertEqual({"cold", "noop", "touch_leaf", "touch_header_top", "touch_header_median"}, set(result["scenarios"]))
        self.assertEqual("include/core.hpp", result["scenarios"]["touch_header_top"]["file"])
        self.assertEqual(2, len(result["scenarios"]["noop"]["runs"]))
        self.assertEqual(5.0, result["scenarios"]["cold"]["configure_s"])
        self.assertEqual("20261005T120000Z", result["id"])
        self.assertEqual("a note", result["note"])
        self.assertEqual(result["fingerprint_id"], fp.fingerprint_id(result["fingerprint"]))
        self.assertEqual(2, result["include_graph"]["objects"])

    def test_without_the_cold_scenario_an_existing_build_is_required(self):
        builder = FakeBuilder()
        self.take(builder, scenarios=("noop",))
        self.assertEqual("ensure_built", builder.calls[0])
        with self.assertRaises(snap.SnapshotError):
            self.take(FakeBuilder(built=False), scenarios=("noop",))

    def test_unknown_scenarios_are_rejected(self):
        with self.assertRaises(snap.SnapshotError):
            self.take(scenarios=("cold", "bogus"))

    def test_targets_are_pinned_from_the_first_snapshot_and_overridable(self):
        first = self.take()
        store.save_snapshot(self.root, first)
        different = FakeBuilder(deps_text=FakeBuilder().deps.replace("core.hpp", "other.hpp"))
        again = self.take(different)
        self.assertEqual("include/core.hpp", again["targets"]["header_top"])   # pinned, although other.hpp is now the top
        overridden = self.take(overrides={"header_top": "include/mid.hpp", "leaf": None})
        self.assertEqual("include/mid.hpp", overridden["targets"]["header_top"])
        self.assertEqual(first["targets"]["leaf"], overridden["targets"]["leaf"])

    def test_scenarios_without_a_suitable_file_are_skipped(self):
        builder = FakeBuilder(deps_text="src/a.o: #deps 1, deps mtime 1 (VALID)\n    ../src/a.cpp\n\n")
        result = self.take(builder, scenarios=("touch_header_top", "touch_header_median"))
        self.assertEqual({}, result["scenarios"])
        self.assertTrue(any("skipping touch_header_top" in line for line in self.log))

    def test_the_aggregate_is_the_median_and_flags_unstable_step_counts(self):
        runs = [
            {"wall_s": 10.0, "steps": 5, "cpu_s": 40.0, "link_s": 1.0, "tail_s": 0.5},
            {"wall_s": 12.0, "steps": 5, "cpu_s": 44.0, "link_s": 1.0, "tail_s": 0.5},
            {"wall_s": 11.0, "steps": 6, "cpu_s": 42.0, "link_s": 1.0, "tail_s": 0.5},
        ]
        result = snap.aggregate(runs)
        self.assertEqual(11.0, result["wall_s"])
        self.assertEqual(42.0, result["cpu_s"])
        self.assertEqual(round(2.0 / 11.0, 3), result["spread"])
        self.assertFalse(result["steps_stable"])
        self.assertTrue(snap.aggregate(runs[:2])["steps_stable"])

    def test_git_state_reads_commit_dirty_flag_and_submodules(self):
        def run(command):
            joined = " ".join(command)
            if "rev-parse HEAD" in joined:
                return "0123456789abcdef\n"
            if "--abbrev-ref" in joined:
                return "feature/x\n"
            if "status --porcelain" in joined:
                return " M src/a.cpp\n"
            if "submodule status" in joined:
                return " 18bdc1ac9c7330fcc4a9200cd8542ca172650728 external/mstd (0.5.1)\n+78742ab68f989e804fa4153ab1ad2f36dbed79da external/devops (x)\n"
            return ""

        state = snap.git_state("/src", run)
        self.assertEqual(("0123456789ab", "feature/x", True), (state["commit"], state["branch"], state["dirty"]))
        self.assertEqual({"external/mstd": "18bdc1ac9c73", "external/devops": "78742ab68f98"}, state["submodules"])


class BuilderEntriesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.build_dir = self.root / "bt"
        self.lines = []

    def tearDown(self):
        self._tmp.cleanup()

    def run_stub(self, command, cwd=None):
        if command[0] == "ninja":
            with open(self.build_dir / ".ninja_log", "a") as handle:
                handle.write("".join(self.lines))
        return 0, "ok"

    def builder(self):
        return snap.Builder(str(self.root), str(self.build_dir), str(self.root / "deps"), "Release", [], "all", 4,
                            run=self.run_stub, clock=iter(range(0, 1000, 5)).__next__)

    def test_the_cold_build_keeps_its_entries_and_later_builds_do_not_replace_them(self):
        builder = self.builder()
        self.lines = [f"0\t2000\t1\t{OBJ_A}\th1\n", f"2000\t2500\t1\tsrc/liblib.so\th2\n"]
        builder.cold()
        self.assertEqual(sorted([OBJ_A, "src/liblib.so"]), sorted(e.output for e in builder.cold_entries))   # a set: no order
        self.lines = [f"0\t500\t2\t{OBJ_B}\th3\n"]
        builder.build()
        self.assertEqual([OBJ_B], [e.output for e in builder.last_steps])
        self.assertEqual(sorted([OBJ_A, "src/liblib.so"]), sorted(e.output for e in builder.cold_entries))   # still the cold build

    def test_all_entries_are_the_newest_per_output_in_log_order(self):
        builder = self.builder()
        builder.claim_build_dir()
        (self.build_dir / ".ninja_log").write_text(f"# ninja log v7\n0\t1000\t1\t{OBJ_A}\th1\n0\t2000\t1\t{OBJ_B}\th2\n5000\t5500\t2\t{OBJ_A}\th3\n")
        self.assertEqual([(OBJ_A, 5500), (OBJ_B, 2000)], [(e.output, e.end) for e in builder.all_entries()])
        self.assertEqual([], self.builder().__class__(str(self.root), str(self.root / "none"), str(self.root / "d"), "Release", [], "all", 1).all_entries())


class ConfigureDirectoryTests(unittest.TestCase):
    def test_cmake_runs_inside_the_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            builder = snap.Builder(directory, str(Path(directory) / "bt"), str(Path(directory) / "deps"), "Release", [], "all", 4,
                                   run=lambda command, cwd=None: calls.append((command[0], cwd)) or (0, "ok"), clock=iter(range(0, 100, 5)).__next__)
            builder.claim_build_dir()
            builder.configure()
        self.assertEqual([("cmake", str(Path(directory) / "bt"))], calls)


class BuilderGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def builder(self, build_dir, **kwargs):
        return snap.Builder(str(self.root), str(build_dir), str(self.root / "deps"), "Release", [], "all", 4, **kwargs)

    def test_a_foreign_directory_is_never_taken_over_or_deleted(self):
        foreign = self.root / "mine"
        foreign.mkdir()
        (foreign / "precious.txt").write_text("keep")
        builder = self.builder(foreign)
        with self.assertRaises(snap.SnapshotError):
            builder.claim_build_dir()
        with self.assertRaises(snap.SnapshotError):
            builder.wipe_build_dir()
        self.assertTrue((foreign / "precious.txt").exists())

    def test_an_empty_or_new_directory_is_claimed_and_can_be_wiped(self):
        target = self.root / "bt"
        builder = self.builder(target)
        builder.claim_build_dir()
        (target / "junk.o").write_text("x")
        builder.wipe_build_dir()
        self.assertEqual([snap.MARKER], os.listdir(target))
        builder.claim_build_dir()   # accepted again

    def test_the_build_counts_only_the_lines_a_build_added(self):
        target = self.root / "bt"
        log = target / ".ninja_log"

        def run(command, cwd=None):
            with open(log, "a") as handle:
                handle.write("0\t2000\t5\tsrc/a.o\tnewhash\n")
            return 0, "ok"

        builder = self.builder(target, run=run, clock=iter([100.0, 107.5]).__next__, probe=lambda: (None, 0.0))
        builder.claim_build_dir()
        log.write_text("# ninja log v7\n0\t1000\t1\tsrc/old.o\toldhash\n")
        result = builder.build()
        self.assertEqual({"steps": 1, "cpu_s": 2.0, "link_s": 0.0, "tail_s": 0.0, "wall_s": 7.5}, result)

    def test_a_failed_build_reports_the_tail_of_the_output(self):
        target = self.root / "bt"
        builder = self.builder(target, run=lambda command, cwd=None: (1, "line\n" * 40 + "error: boom"))
        builder.claim_build_dir()
        with self.assertRaises(snap.SnapshotError) as caught:
            builder.build()
        self.assertIn("error: boom", str(caught.exception))

    def test_defaults_disable_ccache_and_native_tuning(self):
        args = self.builder(self.root / "bt").cmake_args
        for expected in ("-DBUILD_WITH_NATIVE=OFF", "-DCMAKE_CXX_COMPILER_LAUNCHER=", "-DCMAKE_BUILD_TYPE=Release"):
            self.assertIn(expected, args)


class ReportTests(unittest.TestCase):
    def snapshots(self):
        return [
            make_snapshot("20261001T100000Z", wall=100.0, note="baseline state"),
            make_snapshot("20261008T100000Z", wall=90.0, note="<script>alert(1)</script>", steps=3),
            make_snapshot("20261015T100000Z", wall=60.0, commit="deadbeef0000", dirty=True),
        ]

    def test_durations_are_readable(self):
        self.assertEqual("0.42s", report.format_duration(0.42))
        self.assertEqual("12.5s", report.format_duration(12.46))
        self.assertEqual("2m 05.0s", report.format_duration(125.0))
        self.assertEqual("1h 01m", report.format_duration(3660))
        self.assertEqual("-", report.format_duration(None))

    def test_the_page_has_every_panel_the_baseline_and_escaped_notes(self):
        snapshots = self.snapshots()
        page = report.render_html(snapshots, snapshots[0], "abc123abc123")
        self.assertEqual(len(report.WALL_PANELS) + len(report.STEP_PANELS) + len(report.GRAPH_PANELS), page.count('<div class="panel">'))
        self.assertIn('class="baseline"', page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("x86_64 | Test CPU", page)
        self.assertIn("deadbeef0000*", page)

    def test_a_single_snapshot_and_missing_data_still_render(self):
        only = [make_snapshot("20261001T100000Z")]
        page = report.render_html(only, None, "abc123abc123")
        self.assertIn("none yet", page)
        self.assertNotIn('class="baseline"', page)
        empty = {"id": "20261001T100000Z", "created_at": "2026-10-01T10:00:00Z", "fingerprint": fingerprint(), "scenarios": {}}
        self.assertIn("no data", report.render_html([empty], None, "abc123abc123"))

    def test_changed_touch_targets_are_flagged(self):
        a = make_snapshot("20261001T100000Z")
        b = make_snapshot("20261002T100000Z", targets={"leaf": "src/other.cpp"})
        self.assertIn("touched files differ", report.render_html([a, b], None, "abc123abc123"))
        self.assertNotIn("touched files differ", report.render_html([a, a], None, "abc123abc123"))

    def test_points_sit_in_snapshot_order_and_lines_need_two_points(self):
        snapshots = self.snapshots()
        chart = report.svg_chart("t", snapshots, lambda s: report.wall_value(s, "cold"), lambda s, v: str(v), None, report.format_duration)
        self.assertEqual(3, chart.count("<circle"))
        self.assertEqual(1, chart.count("<polyline"))
        single = report.svg_chart("t", snapshots[:1], lambda s: report.wall_value(s, "cold"), lambda s, v: str(v), None, report.format_duration)
        self.assertEqual(0, single.count("<polyline"))

    def test_the_table_shows_the_change_against_the_baseline(self):
        snapshots = self.snapshots()
        table = report.render_table(snapshots, snapshots[0])
        lines = table.splitlines()
        self.assertIn("1m 40.0s", lines[2])   # the baseline row itself has no delta
        self.assertNotIn("%", lines[2])
        self.assertIn("(-10%)", lines[3])
        self.assertIn("(-40%)", lines[4])
        self.assertIn("deadbeef*", lines[4])
        self.assertIn("baseline state", lines[2])


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        store.save_snapshot(self.root, make_snapshot("20261001T100000Z", wall=100.0))
        store.save_snapshot(self.root, make_snapshot("20261002T100000Z", wall=80.0))

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = pqbt.main(["--data-dir", str(self.root), *args])
        return code, output.getvalue()

    def test_baseline_set_show_and_list(self):
        self.assertEqual("no baseline", self.run_cli("baseline", "show")[1].strip())
        self.run_cli("baseline", "set", "20261001T100000Z")
        self.assertEqual("20261001T100000Z", self.run_cli("baseline", "show")[1].strip())
        listing = self.run_cli("list")[1]
        self.assertIn("abc123abc123", listing)
        self.assertIn("2 snapshots", listing)
        self.assertIn("baseline 20261001T100000Z", listing)

    def test_baseline_latest_and_unknown_snapshot(self):
        self.run_cli("baseline", "set")
        self.assertEqual("20261002T100000Z", self.run_cli("baseline", "show")[1].strip())
        with self.assertRaises(SystemExit):
            self.run_cli("baseline", "set", "nope")

    def test_report_writes_the_page_and_prints_the_table(self):
        self.run_cli("baseline", "set", "20261001T100000Z")
        code, text = self.run_cli("report")
        self.assertEqual(0, code)
        self.assertIn("(-20%)", text)
        page = (self.root / "report-abc123abc123.html").read_text()
        self.assertIn("<svg", page)
        out = self.root / "custom" / "x.html"
        self.run_cli("report", "--out", str(out), "--fingerprint", "abc123abc123")
        self.assertTrue(out.exists())

    def test_report_without_html_and_with_an_empty_store(self):
        self.run_cli("report", "--no-html")
        self.assertEqual([], list(self.root.glob("report-*.html")))
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(SystemExit):
            pqbt.main(["--data-dir", empty, "report"])

    def test_a_busy_machine_refuses_to_measure(self):
        # a regression of the guard must fail this test, not start a real build
        with (
            mock.patch.object(os, "getloadavg", return_value=(30.0, 20.0, 10.0)),
            mock.patch.object(snap, "take_snapshot", side_effect=AssertionError("must not measure a busy machine")),
            self.assertRaises(SystemExit) as caught,
        ):
            pqbt.main(["--data-dir", str(self.root), "snapshot"])
        self.assertIn("busy", str(caught.exception))



def scenario(wall, steps, spread=0.01, runs=3, file=None, cpu=None, stable=True):
    data = {"wall_s": wall, "steps": steps, "cpu_s": cpu if cpu is not None else wall * 4, "spread": spread,
            "runs": [{"wall_s": wall}] * runs, "steps_stable": stable}
    if file:
        data["file"] = file
    return data


def compared_snapshot(ident, scenarios, fid="abc123abc123", pairs=1000, digest="d" * 64, load=0.1, **fingerprint_changes):
    snapshot = make_snapshot(ident, fid=fid)
    snapshot["scenarios"] = scenarios
    snapshot["include_graph"] = {"objects": 10, "unique_files": 50, "include_pairs": pairs, "project_files": 20,
                                 "project_pairs": pairs // 2, "digest": digest}
    snapshot["load"] = {"start": load, "end": load}
    snapshot["fingerprint"] = fingerprint(**fingerprint_changes)
    return snapshot


class JudgeTests(unittest.TestCase):
    def verdict(self, steps_a, steps_b, wall_a, wall_b, threshold=0.05):
        return compare.judge(steps_a, steps_b, wall_a, wall_b, threshold)

    def test_every_combination_of_steps_and_timing(self):
        self.assertEqual(("faster", "improved"), self.verdict(40, 20, 4.0, 2.0))
        self.assertEqual(("slower", "regressed"), self.verdict(20, 40, 2.0, 4.0))
        self.assertEqual(("same", "fewer steps, time within noise"), self.verdict(40, 20, 4.0, 3.95))
        self.assertEqual(("same", "more steps, time within noise"), self.verdict(20, 40, 4.0, 4.05))
        self.assertEqual(("same", "unchanged"), self.verdict(40, 40, 4.0, 4.05))
        self.assertEqual("faster with the same steps: check noise", self.verdict(40, 40, 4.0, 2.0)[1])
        self.assertEqual("slower with the same steps: check noise", self.verdict(40, 40, 2.0, 4.0)[1])
        self.assertEqual("more steps but faster: check noise", self.verdict(20, 40, 4.0, 2.0)[1])
        self.assertEqual("fewer steps but slower: check noise", self.verdict(40, 20, 2.0, 4.0)[1])

    def test_a_change_must_exceed_the_relative_threshold_and_the_absolute_floor(self):
        self.assertEqual("same", self.verdict(5, 5, 10.0, 10.9, threshold=0.10)[0])   # +9% under a 10% threshold
        self.assertEqual("slower", self.verdict(5, 5, 10.0, 11.1, threshold=0.10)[0])
        self.assertEqual("same", self.verdict(0, 0, 0.01, 0.05)[0])   # a no-op build: 0.04 s is below the 0.05 s floor
        self.assertEqual("slower", self.verdict(0, 0, 0.01, 0.2)[0])


class NoiseTests(unittest.TestCase):
    def test_the_threshold_follows_twice_the_larger_spread_with_a_floor(self):
        self.assertEqual((0.05, False), compare.noise_threshold(scenario(1, 1, spread=0.01), scenario(1, 1, spread=0.02)))
        self.assertEqual((0.30, False), compare.noise_threshold(scenario(1, 1, spread=0.15), scenario(1, 1, spread=0.01)))

    def test_a_single_run_widens_the_floor(self):
        self.assertEqual((0.10, True), compare.noise_threshold(scenario(1, 1, spread=0.0, runs=1), scenario(1, 1, spread=0.01)))


class CompareTests(unittest.TestCase):
    def pair(self, a_scenarios, b_scenarios, **kwargs):
        return (compared_snapshot("20261001T100000Z", a_scenarios, **kwargs.get("a", {})),
                compared_snapshot("20261002T100000Z", b_scenarios, **kwargs.get("b", {})))

    def test_different_fingerprints_are_refused_and_the_difference_is_named(self):
        a = compared_snapshot("20261001T100000Z", {}, fid="aaaaaaaaaaaa")
        b = compared_snapshot("20261002T100000Z", {}, fid="bbbbbbbbbbbb", compiler="clang 20", jobs=4)
        with self.assertRaises(compare.CompareError) as caught:
            compare.compare(a, b)
        message = str(caught.exception)
        self.assertIn("aaaaaaaaaaaa vs bbbbbbbbbbbb", message)
        self.assertIn("compiler: 'g++ 15' vs 'clang 20'", message)
        self.assertIn("jobs: 8 vs 4", message)

    def test_rows_follow_the_scenario_order_and_skip_missing_ones(self):
        a, b = self.pair(
            {"touch_leaf": scenario(1.0, 6), "cold": scenario(100.0, 500, runs=1), "noop": scenario(0.01, 0)},
            {"touch_leaf": scenario(1.0, 6), "cold": scenario(100.0, 500, runs=1)},
        )
        self.assertEqual(["cold", "touch_leaf"], [row["scenario"] for row in compare.compare(a, b)["rows"]])

    def test_an_improvement_is_reported_with_its_changes(self):
        a, b = self.pair({"touch_header_top": scenario(4.0, 40, file="h.hpp")}, {"touch_header_top": scenario(2.0, 20, file="h.hpp")})
        row = compare.compare(a, b)["rows"][0]
        self.assertEqual("improved", row["verdict"])
        self.assertAlmostEqual(-0.5, row["wall_change"])
        self.assertAlmostEqual(-0.5, row["steps_change"])

    def test_scenarios_that_touched_different_files_are_not_compared(self):
        a, b = self.pair({"touch_leaf": scenario(1.0, 6, file="src/a.cpp")}, {"touch_leaf": scenario(9.0, 60, file="src/b.cpp")})
        row = compare.compare(a, b)["rows"][0]
        self.assertFalse(row["comparable"])
        self.assertIn("src/a.cpp vs src/b.cpp", row["verdict"])
        self.assertNotIn("wall_change", row)

    def test_the_include_graph_changes_and_digest_are_reported(self):
        a, b = self.pair({}, {}, a={"pairs": 1000}, b={"pairs": 900, "digest": "e" * 64})
        result = compare.compare(a, b)
        pairs = next(row for row in result["graph"] if row["key"] == "include_pairs")
        self.assertAlmostEqual(-0.1, pairs["change"])
        self.assertTrue(result["digest_changed"])
        self.assertFalse(compare.compare(*self.pair({}, {}))["digest_changed"])

    def test_a_busy_start_is_flagged(self):
        a, b = self.pair({}, {}, a={"load": 9.0})
        self.assertEqual([a["id"]], compare.compare(a, b)["busy"])
        self.assertEqual([], compare.compare(*self.pair({}, {}))["busy"])

    def test_the_text_has_the_verdicts_the_noise_and_the_warnings(self):
        a, b = self.pair(
            {"cold": scenario(100.0, 500, runs=1), "touch_header_top": scenario(4.0, 40, file="h.hpp", stable=False)},
            {"cold": scenario(80.0, 500, runs=1), "touch_header_top": scenario(2.0, 20, file="h.hpp")},
            a={"load": 9.0},
        )
        text = compare.render(compare.compare(a, b))
        for expected in ("before: 20261001T100000Z", "after : 20261002T100000Z", "improved", "-50.0%", "+-10%*", "identical",
                         "single run per scenario", "was busy", "number of steps varied", "touch top header"):
            self.assertIn(expected, text)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = [make_snapshot(f"2026100{i}T100000Z") for i in (1, 2, 3)]
        self.baseline = self.snapshots[0]

    def resolve(self, reference, baseline="default"):
        return compare.resolve(self.snapshots, self.baseline if baseline == "default" else baseline, reference)

    def test_symbolic_references(self):
        self.assertEqual("20261003T100000Z", self.resolve("latest")["id"])
        self.assertEqual("20261002T100000Z", self.resolve("previous")["id"])
        self.assertEqual("20261001T100000Z", self.resolve("baseline")["id"])

    def test_ids_and_unique_prefixes(self):
        self.assertEqual("20261002T100000Z", self.resolve("20261002T100000Z")["id"])
        self.assertEqual("20261002T100000Z", self.resolve("20261002")["id"])

    def test_errors(self):
        for reference in ("2026", "nope"):
            with self.assertRaises(compare.CompareError):
                self.resolve(reference)
        with self.assertRaises(compare.CompareError):
            self.resolve("baseline", baseline=None)
        with self.assertRaises(compare.CompareError):
            compare.resolve(self.snapshots[:1], None, "previous")
        with self.assertRaises(compare.CompareError):
            compare.resolve([], None, "latest")


class CompareCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for ident, wall in (("20261001T100000Z", 4.0), ("20261002T100000Z", 3.0), ("20261003T100000Z", 2.0)):
            store.save_snapshot(self.root, compared_snapshot(ident, {"touch_leaf": scenario(wall, 6, file="a.cpp")}))
        store.save_snapshot(self.root, compared_snapshot("20261004T100000Z", {}, fid="bbbbbbbbbbbb", compiler="clang 20"))

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pqbt.main(["--data-dir", str(self.root), "compare", "--fingerprint", "abc123abc123", *args])
        return output.getvalue()

    def test_without_a_baseline_the_latest_is_compared_with_the_previous(self):
        text = self.run_cli()
        self.assertIn("before: 20261002T100000Z", text)
        self.assertIn("after : 20261003T100000Z", text)

    def test_with_a_baseline_the_latest_is_compared_with_it(self):
        store.set_baseline(self.root, "abc123abc123", "20261001T100000Z")
        self.assertIn("before: 20261001T100000Z", self.run_cli())

    def test_the_baseline_is_not_compared_with_itself(self):
        store.set_baseline(self.root, "abc123abc123", "20261003T100000Z")
        self.assertIn("before: 20261002T100000Z", self.run_cli())

    def test_one_argument_is_compared_with_the_latest_and_two_arguments_explicitly(self):
        self.assertIn("before: 20261001T100000Z", self.run_cli("20261001"))
        text = self.run_cli("20261001", "20261002")
        self.assertIn("before: 20261001T100000Z", text)
        self.assertIn("after : 20261002T100000Z", text)

    def test_other_fingerprints_and_the_same_snapshot_are_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("20261001T100000Z", "20261004T100000Z")
        self.assertIn("different fingerprints", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("20261001T100000Z", "20261001T100000Z")
        self.assertIn("same snapshot", str(caught.exception))



ROOT, BUILD = "/src", "/src/build"
OBJ_G = "external/gt/CMakeFiles/gt.dir/src/g.cc.o"
PCH = "src/CMakeFiles/pq_pch.dir/cmake_pch.hxx.gch"
DETAIL_ENTRIES = [
    entry(1000, 3000, PCH),
    entry(3000, 7000, OBJ_A),
    entry(3000, 6000, OBJ_B),
    entry(3000, 5000, OBJ_G),
    entry(7000, 7500, "src/liblib.so"),
    entry(7500, 7600, "apps/PQ"),
]
DETAIL_DEPS = {
    OBJ_A: ["../src/a.cpp", "../include/core.hpp", "../include/mid.hpp", "/usr/include/stdio.h", "../external/mstd/include/m.hpp"],
    OBJ_B: ["../src/b.cpp", "../include/core.hpp", "gen/generated.hpp"],
    OBJ_G: ["../external/gt/src/g.cc", "/usr/include/stdio.h"],
}
DETAIL_CHURN = {"include/core.hpp": 3, "include/mid.hpp": 1, "external/mstd": 2, "src/a.cpp": 4, "src/b.cpp": 0, "external/gt/src/g.cc": 9}


def sample_detail(touches=None, churn=DETAIL_CHURN, source="cold", entries=DETAIL_ENTRIES, trace=None):
    return detail.make_detail("20261006T100000Z", "abc123abc123", 8, source, entries, DETAIL_DEPS, touches or {}, ROOT, BUILD,
                              churn, "origin/dev" if churn is not None else None, 180, trace)


def sample_snapshot(note=""):
    snapshot = make_snapshot("20261006T100000Z", note=note)
    snapshot["fingerprint"] = fingerprint()
    return snapshot


class NinjaDetailHelperTests(unittest.TestCase):
    def test_outputs_are_classified(self):
        expected = {
            "src/molsys/CMakeFiles/molsys.dir/a.cpp.o": "compile", "x.obj": "compile", PCH: "pch", "src/libm.so": "library",
            "lib/libm.so.1.2": "library", "libm.a": "library", "apps/PQ": "executable", "tests/t/testX": "executable",
            "CMakeFiles/gen.util": "other", "build.ninja": "other", "tests/CMakeFiles/cleanup_gcda": "other",
        }
        for output, kind in expected.items():
            self.assertEqual(kind, ninja.kind_of_output(output), output)

    def test_the_source_of_an_object_follows_the_mirrored_tree(self):
        self.assertEqual("src/molsys/a.cpp", ninja.source_of_object("src/molsys/CMakeFiles/molsys.dir/a.cpp.o"))
        self.assertEqual("tests/src/t/x.cpp", ninja.source_of_object("tests/src/t/CMakeFiles/x.dir/x.cpp.o"))
        self.assertEqual("external/gt/src/g.cc", ninja.source_of_object(OBJ_G))
        self.assertEqual("y.cpp", ninja.source_of_object("CMakeFiles/x.dir/y.cpp.o"))
        self.assertIsNone(ninja.source_of_object("src/liblib.so"))

    def test_repo_relative_paths(self):
        self.assertEqual(("src/a.cpp", False), ninja.repo_relative("../src/a.cpp", ROOT, BUILD))
        self.assertEqual(("include/x.hpp", False), ninja.repo_relative("/src/include/x.hpp", ROOT, BUILD))
        self.assertEqual(("external/mstd/m.hpp", True), ninja.repo_relative("../external/mstd/m.hpp", ROOT, BUILD))
        self.assertIsNone(ninja.repo_relative("/usr/include/stdio.h", ROOT, BUILD))
        self.assertIsNone(ninja.repo_relative("gen/generated.hpp", ROOT, BUILD))

    def test_log_entries_in_order_and_deduplication(self):
        text = "# ninja log v7\n0\t1000\t1\ta.o\th1\n0\t2000\t1\tb.o\th2\n5000\t5500\t2\ta.o\th3\n3000\t4000\t1\tl.so\th4\n3000\t4000\t1\tl.so.1\th4\n"
        entries = ninja.parse_log(ninja.ordered_lines(text))
        self.assertEqual(["a.o", "b.o", "a.o", "l.so", "l.so.1"], [e.output for e in entries])
        self.assertEqual([("a.o", 5500), ("b.o", 2000), ("l.so", 4000), ("l.so.1", 4000)],
                         [(e.output, e.end) for e in ninja.latest_per_output(entries)])   # the newer a.o replaces the older one in place
        self.assertEqual(4, len(ninja.unique_steps(entries)))   # the two outputs of one command are one step


class DetailPartsTests(unittest.TestCase):
    def test_modules_and_groups(self):
        expected = {
            "src/molsys/a.cpp": ("src/molsys", "src (project)"), "src/main.cpp": ("src", "src (project)"),
            "tests/src/t/x.cpp": ("tests", "tests"), "benchmarks/perf/p.cpp": ("perf", "perf"), "benchmarks/src/b.cpp": ("benchmarks", "benchmarks"),
            "apps/PQ.cpp": ("apps", "apps"), "external/gt/g.cc": ("third-party", "third-party"), "_deps/eigen/e.cpp": ("third-party", "third-party"),
            "other/x.cpp": ("other", "other"), None: ("other", "other"),
        }
        for source, (module, group) in expected.items():
            self.assertEqual((module, group), (detail.module_of(source), detail.group_of(detail.module_of(source))), source)

    def test_steps_have_kinds_offsets_and_include_counts(self):
        steps = detail.build_steps(DETAIL_ENTRIES, DETAIL_DEPS, ROOT, BUILD, DETAIL_CHURN)
        self.assertEqual([PCH, OBJ_G, OBJ_A, OBJ_B, "src/liblib.so", "apps/PQ"], [s["o"] for s in steps])   # by start, then name
        self.assertEqual(["pch", "compile", "compile", "compile", "library", "executable"], [s["k"] for s in steps])
        self.assertEqual((0.0, 2.0, 2.0), (steps[0]["a"], steps[0]["e"], steps[0]["s"]))   # offsets are relative to the first start
        a, b, third_party = steps[2], steps[3], steps[1]
        self.assertEqual((2.0, 6.0, 4.0, "src/a.cpp"), (a["a"], a["e"], a["s"], a["src"]))
        self.assertEqual((5, 3), (a["n"], a["p"]))   # 5 files in all; 3 repository files besides the source itself
        self.assertEqual((3, 1), (b["n"], b["p"]))   # generated and system files are not repository files
        self.assertEqual(4, a["ch"])
        self.assertEqual(0, b["ch"])
        self.assertNotIn("ch", third_party)   # third-party sources are not ours to change

    def test_a_file_listed_twice_in_the_dependencies_counts_once(self):
        deps = {OBJ_A: ["../src/a.cpp", "../include/x.hpp", "../include/x.hpp", "/src/include/x.hpp"]}
        step = detail.build_steps([entry(0, 1000, OBJ_A)], deps, ROOT, BUILD, None)[0]
        self.assertEqual((3, 1), (step["n"], step["p"]))   # three distinct paths; one repository file besides the source

    def test_without_churn_or_deps_those_fields_are_absent(self):
        steps = detail.build_steps(DETAIL_ENTRIES, {}, ROOT, BUILD, None)
        self.assertTrue(all("ch" not in s and "n" not in s and "p" not in s for s in steps))
        self.assertEqual([], detail.build_steps([], {}, ROOT, BUILD, None))

    def test_header_costs_sum_the_compile_time_of_the_includers(self):
        objects, fan_in, cpu, external, sets = detail.header_costs(DETAIL_DEPS, {OBJ_A: 4.0, OBJ_B: 3.0}, ROOT, BUILD)
        self.assertEqual(2, objects)   # OBJ_G has no time, so it is not counted
        self.assertEqual({"include/core.hpp": 2, "include/mid.hpp": 1, "external/mstd/include/m.hpp": 1}, dict(fan_in))
        self.assertEqual({"include/core.hpp": 7.0, "include/mid.hpp": 4.0, "external/mstd/include/m.hpp": 4.0}, dict(cpu))
        self.assertEqual({"include/core.hpp": False, "include/mid.hpp": False, "external/mstd/include/m.hpp": True}, external)
        self.assertEqual(sets["include/mid.hpp"], sets["external/mstd/include/m.hpp"])   # both only included by OBJ_A
        self.assertNotEqual(sets["include/core.hpp"], sets["include/mid.hpp"])

    def test_a_header_included_twice_by_one_object_counts_once(self):
        deps = {OBJ_A: ["../include/x.hpp", "../include/x.hpp", "/src/include/x.hpp"]}
        _, fan_in, cpu, _, _ = detail.header_costs(deps, {OBJ_A: 2.0}, ROOT, BUILD)
        self.assertEqual((1, 2.0), (fan_in["include/x.hpp"], cpu["include/x.hpp"]))

    def test_concurrency_averages_the_running_steps_per_tenth(self):
        steps = [{"a": 0.0, "e": 10.0}, {"a": 0.0, "e": 5.0}]
        self.assertEqual([2.0, 1.0], detail.concurrency(steps, buckets=2))
        three = [{"a": 0.0, "e": 6.0}, {"a": 3.0, "e": 6.0}, {"a": 0.0, "e": 3.0}]
        self.assertEqual([2.0, 2.0], detail.concurrency(three, buckets=2))   # (3 + 3) / 3 s in each half
        self.assertEqual([], detail.concurrency([]))
        self.assertEqual(10, len(detail.concurrency(steps)))

    def test_concentration_of_the_compile_time(self):
        compiles = [{"s": 5.0}, {"s": 3.0}, {"s": 1.0}, {"s": 1.0}]
        self.assertEqual((1, 2, 4), detail.concentration(compiles))   # 5 of 10 is half; 8 of 10 is four fifths
        self.assertEqual((0, 0, 0), detail.concentration([]))


class GitChurnTests(unittest.TestCase):
    def test_changes_are_counted_per_file_once_per_change(self):
        log = "\x01\nsrc/a.cpp\ninclude/x.hpp\nsrc/a.cpp\n\x01\nsrc/a.cpp\n\x01\n\n"

        def run(command):
            return "abc\n" if "rev-parse" in command else log

        counts, ref = detail.git_churn("/r", 90, run)
        self.assertEqual(({"src/a.cpp": 2, "include/x.hpp": 1}, "origin/dev"), (dict(counts), ref))

    def test_a_missing_ref_falls_back_and_no_history_gives_none(self):
        asked = []

        def run(command):
            if "rev-parse" in command:
                asked.append(command[-1])
                return "abc\n" if command[-1].startswith("HEAD") else ""
            return "\x01\nf\n"

        self.assertEqual("HEAD", detail.git_churn("/r", 90, run)[1])
        self.assertEqual(["origin/dev^{commit}", "dev^{commit}", "HEAD^{commit}"], asked)
        self.assertEqual((None, None), detail.git_churn("/r", 90, lambda command: ""))

    def test_the_window_is_part_of_the_command(self):
        commands = []
        detail.git_churn("/r", 42, lambda command: commands.append(command) or "x\n")
        self.assertTrue(any("--since=42.days" in command for command in commands))


class MakeDetailTests(unittest.TestCase):
    def test_headers_are_ranked_by_rebuild_cpu_and_carry_their_churn(self):
        headers = sample_detail()["headers"]
        self.assertEqual(["include/core.hpp", "external/mstd/include/m.hpp", "include/mid.hpp"], [h["f"] for h in headers])
        self.assertEqual([(2, 7.0, 3), (1, 4.0, 2), (1, 4.0, 1)], [(h["fan_in"], h["cpu_s"], h["ch"]) for h in headers])   # m.hpp: 2 submodule bumps
        self.assertEqual([False, True, False], [h["x"] for h in headers])

    def test_touches_keep_their_rebuilt_steps_by_time(self):
        touches = {"touch_leaf": {"file": "src/a.cpp", "wall_s": 1.5, "entries": [entry(1500, 1800, "src/liblib.so"), entry(0, 1500, OBJ_A)]}}
        touch = sample_detail(touches)["touch"]["touch_leaf"]
        self.assertEqual(("src/a.cpp", 1.5), (touch["file"], touch["wall_s"]))
        self.assertEqual([(OBJ_A, "compile", 1.5), ("src/liblib.so", "library", 0.3)], [(s["o"], s["k"], s["s"]) for s in touch["steps"]])

    def test_without_git_history_there_is_no_churn(self):
        built = sample_detail(churn=None)
        self.assertTrue(all("ch" not in h for h in built["headers"]))
        self.assertIsNone(built["churn_days"])
        self.assertEqual(("local-build-detail", 1, "cold", 8, 3), (built["kind"], built["schema_version"], built["source"], built["jobs"], built["objects"]))
        self.assertEqual({"include/core.hpp": 2, "include/mid.hpp": 1, "external/mstd/include/m.hpp": 1}, built["fan_in"])   # every header, not only the top list

    def test_at_most_the_top_headers_are_stored(self):
        deps = {OBJ_A: [f"../include/h{i}.hpp" for i in range(detail.TOP_HEADERS_STORED + 25)]}
        built = detail.make_detail("i", "f", 1, "cold", [entry(0, 1000, OBJ_A)], deps, {}, ROOT, BUILD, None, None)
        self.assertEqual(detail.TOP_HEADERS_STORED, len(built["headers"]))
        self.assertEqual(detail.TOP_HEADERS_STORED + 25, len(built["fan_in"]))   # the fan-in of every header is kept


class DetailRenderTests(unittest.TestCase):
    def render(self, built=None, top=12, **kwargs):
        return detail.render(built or sample_detail(**kwargs), sample_snapshot("a note"), top)

    def test_a_cold_report_has_every_section_and_the_timeline(self):
        text = self.render()
        for expected in ("== Build at a glance ==", "== Where the CPU goes ==", "== Slowest compile steps ==",
                         "== Headers: what a change rebuilds ==", "== What small changes rebuild ==", "== Linking ==",
                         'a note', "cold build: wall 6.6s, CPU 11.6s, 6 steps (3 compiles), 8 parallel jobs",
                         "of the 8 job slots", "idle cores", "precompiled header pq_pch: 2.0s, ready after 2.0s"):
            self.assertIn(expected, text)

    def test_a_report_from_the_log_has_no_timeline(self):
        text = self.render(source="log")
        self.assertIn("no timeline", text)
        self.assertNotIn("parallelism:", text)
        self.assertNotIn("precompiled header pq_pch", text)
        self.assertNotIn("ran ", text.split("== Slowest compile steps ==")[1].split("\n")[1])   # no "ran" column in the table header
        self.assertIn("ran", self.render().split("== Slowest compile steps ==")[1].split("\n")[1])

    def test_what_limits_the_wall_time_follows_the_busy_slots(self):
        self.assertIn("CPU throughput: the cores", detail.bound_by(0.9))
        self.assertIn("mostly CPU throughput", detail.bound_by(0.7))
        self.assertIn("idle cores", detail.bound_by(0.3))

    def test_the_cpu_is_split_by_group_and_module(self):
        text = self.render()
        groups = text.split("== Where the CPU goes ==")[1].split("==")[0]
        self.assertRegex(groups, r"src \(project\)\s+2\s+7\.0s\s+60\.3%")   # 4.0 s + 3.0 s of 11.6 s
        self.assertIn("  src", groups)
        self.assertIn("third-party", groups)
        self.assertIn("precompiled header", groups)
        self.assertIn("linking libraries", groups)
        self.assertIn("linking executables", groups)

    def test_the_slowest_steps_show_includes_and_changes(self):
        section = self.render().split("== Slowest compile steps ==")[1].split("== Headers")[0]
        rows = [line.split() for line in section.splitlines() if line.rstrip().endswith(".cpp") or line.rstrip().endswith(".cc")]
        # time, share of the CPU, when it ran (ends in the last tenth: *), repository files, PRs, file
        self.assertEqual(["4.0s", "34.5%", "2-6s*", "3", "4", "src/a.cpp"], rows[0])
        self.assertEqual("src/b.cpp", rows[1][-1])
        self.assertEqual(["2.0s", "17.2%", "2-4s", "0", "-", "external/gt/src/g.cc"], rows[2])   # no repository files, no PR count
        self.assertIn("concentration: 2 files (66.7%) make up half of the compile CPU, 3 files (100.0%) four fifths", section)   # 4+3 of 9 s

    def test_the_burden_ranks_by_changes_times_rebuild_and_marks_submodules(self):
        # mid.hpp is cheap to rebuild (4.0 s) but changed in 10 PRs: 40 s, ahead of core.hpp (3 x 7.0 = 21 s) and m.hpp (2 x 4.0 = 8 s)
        churn = {"include/core.hpp": 3, "include/mid.hpp": 10, "external/mstd": 2}
        text = self.render(churn=churn)
        section = text.split("By burden:")[1].split("By rebuild cost")[0]
        order = [next(token for token in line.split() if token.endswith(".hpp")) for line in section.splitlines() if ".hpp" in line]
        self.assertEqual(["include/mid.hpp", "include/core.hpp", "external/mstd/include/m.hpp"], order)
        self.assertIn("(external)", section)
        self.assertIn("40.0s", section)
        cost = text.split("By rebuild cost")[1].split("==")[0]
        cost_order = [next(token for token in line.split() if token.endswith(".hpp")) for line in cost.splitlines() if ".hpp" in line]
        self.assertEqual(["include/core.hpp", "external/mstd/include/m.hpp"], cost_order)   # by cost alone, mid.hpp is not first...
        self.assertIn("+1 with the same includers: mid.hpp", cost)                           # ...it is folded into m.hpp's row
        self.assertIn("21.0s", section)   # core.hpp: 3 PRs x 7.0 s

    def test_headers_with_the_same_includers_are_shown_once(self):
        deps = {OBJ_A: ["../include/x.hpp", "../include/x.tpp", "../include/y.hpp"], OBJ_B: ["../include/x.hpp", "../include/x.tpp"]}
        built = detail.make_detail("i", "f", 4, "cold", [entry(0, 4000, OBJ_A), entry(0, 3000, OBJ_B)], deps, {}, ROOT, BUILD, None, None)
        text = detail.render(built, sample_snapshot(), 12)
        cost = text.split("By rebuild cost")[1].split("==")[0]
        self.assertIn("include/x.hpp  +1 with the same includers: x.tpp", cost)
        self.assertEqual(1, cost.count("x.tpp"))
        self.assertIn("include/y.hpp", cost)

    def test_without_churn_only_the_cost_list_is_shown(self):
        text = self.render(churn=None)
        self.assertNotIn("By burden", text)
        self.assertIn("By rebuild cost", text)

    def test_relinking_is_judged_by_cpu_not_by_the_number_of_steps(self):
        many_small_links = [entry(0, 3000, OBJ_A)] + [entry(3000 + i, 3010 + i, f"tests/t{i}") for i in range(10)]
        heavy_compile = {"touch_header_median": {"file": "h.hpp", "wall_s": 3.0, "entries": many_small_links}}
        self.assertNotIn("mostly relinking", detail.render(sample_detail(heavy_compile), sample_snapshot(), 12))
        all_links = [entry(0, 100, OBJ_A)] + [entry(100, 600, f"tests/t{i}") for i in range(4)]
        light_compile = {"touch_leaf": {"file": "a.cpp", "wall_s": 0.6, "entries": all_links}}
        text = detail.render(sample_detail(light_compile), sample_snapshot(), 12)
        self.assertIn("mostly relinking (95.2% of the CPU): 0 libraries and 4 executables", text)

    def test_touch_scenarios_report_steps_by_kind(self):
        touches = {"touch_leaf": {"file": "src/a.cpp", "wall_s": 1.5, "entries": [entry(0, 1500, OBJ_A), entry(1500, 1800, "src/liblib.so")]}}
        text = detail.render(sample_detail(touches), sample_snapshot(), 12)
        self.assertIn("touching a source file: src/a.cpp", text)
        self.assertIn("2 steps, wall 1.5s, CPU 1.8s: compile 1 (1.5s), library 1 (0.3s)", text)
        self.assertIn("no touch scenarios", self.render())

    def test_top_limits_the_rows(self):
        deps = {OBJ_A: [f"../include/h{i}.hpp" for i in range(30)]}
        built = detail.make_detail("i", "f", 1, "cold", [entry(0, 1000, OBJ_A)], deps, {}, ROOT, BUILD, None, None)
        text = detail.render(built, sample_snapshot(), 5)
        cost = text.split("By rebuild cost")[1].split("==")[0]
        self.assertEqual(1, cost.count("with the same includers"))   # one group of 30 identical headers, one row
        # two sets of headers with the same fan-in and CPU but different includers must stay two rows, and top=1 keeps one
        built = detail.make_detail("i", "f", 1, "cold", [entry(0, 1000, OBJ_A), entry(0, 1000, OBJ_B)],
                                   {OBJ_A: [f"../include/a{i}.hpp" for i in range(8)], OBJ_B: [f"../include/b{i}.hpp" for i in range(8)]},
                                   {}, ROOT, BUILD, None, None)
        for top, rows in ((2, 2), (1, 1)):
            cost = detail.render(built, sample_snapshot(), top).split("By rebuild cost")[1].split("==")[0]
            self.assertEqual(rows, cost.count("with the same includers"))

    def test_a_snapshot_without_steps_says_so(self):
        built = sample_detail(entries=[])
        self.assertIn("no build steps", detail.render(built, sample_snapshot(), 12))

    def test_the_table_helper_aligns_numbers_right_and_the_last_column_left(self):
        text = detail.table([["a", "b", "name"], [1, 22, "x"], [333, 4, "yy"]], left=())
        self.assertEqual(["  a   b  name", "---  --  ----", "  1  22  x", "333   4  yy"], text.splitlines())


class DetailStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_detail_round_trips_and_is_not_a_snapshot(self):
        built = sample_detail()
        path = store.save_detail(self.root, built)
        self.assertEqual(self.root / "details" / "20261006T100000Z.json", path)
        self.assertEqual(built, store.load_detail(self.root, "20261006T100000Z"))
        self.assertEqual([], store.load_snapshots(self.root))   # details never show up as snapshots

    def test_missing_damaged_and_foreign_details_are_none(self):
        self.assertIsNone(store.load_detail(self.root, "nope"))
        (self.root / "details").mkdir()
        (self.root / "details" / "bad.json").write_text("{broken")
        (self.root / "details" / "foreign.json").write_text('{"kind": "x", "schema_version": 1}')
        (self.root / "details" / "old.json").write_text('{"kind": "local-build-detail", "schema_version": 9}')
        for name in ("bad", "foreign", "old"):
            self.assertIsNone(store.load_detail(self.root, name))


class SnapshotDetailTests(SnapshotTests):
    def test_the_detail_comes_from_the_cold_build_and_the_first_touch_run(self):
        result = self.take()
        self.assertEqual(result["id"], self.detail["id"])
        self.assertEqual("cold", self.detail["source"])
        self.assertEqual(len(COLD_ENTRIES), len(self.detail["steps"]))
        self.assertEqual({"touch_leaf", "touch_header_top", "touch_header_median"}, set(self.detail["touch"]))
        self.assertEqual([OBJ_A, "src/liblib.so"], [s["o"] for s in self.detail["touch"]["touch_leaf"]["steps"]])
        self.assertEqual("src/a.cpp", self.detail["touch"]["touch_leaf"]["file"])
        self.assertEqual(8, self.detail["jobs"])
        self.assertEqual(2, self.detail["objects"])

    def test_the_touch_detail_is_the_first_repetition_not_a_later_one(self):
        self.take(repeat=3)
        self.assertEqual([OBJ_A, "src/liblib.so"], [s["o"] for s in self.detail["touch"]["touch_header_top"]["steps"]])

    def test_without_the_cold_scenario_the_times_come_from_the_log(self):
        builder = FakeBuilder()
        self.take(builder, scenarios=("noop",))
        self.assertEqual("log", self.detail["source"])
        self.assertEqual({}, self.detail["touch"])

    def test_the_noop_scenario_leaves_no_steps_behind(self):
        builder = FakeBuilder()
        self.take(builder, scenarios=("cold", "noop"))
        self.assertEqual([], builder.last_steps)
        self.assertEqual({}, self.detail["touch"])


class SubmoduleProblemTests(unittest.TestCase):
    def run_stub(self, status, used=("external/mstd",)):
        def run(command):
            if "submodule" in command:
                return status
            return "CMakeLists.txt\n" if command[command.index("-F") + 1] in used else ""

        return run

    def test_stale_missing_and_conflicted_submodules_are_reported(self):
        status = (" aaa111 external/ok (v1)\n+bbb222 external/mstd (v2)\n-ccc333 external/devops\nUddd444 external/gt (v3)\n")
        problems = snap.submodule_problems("/r", self.run_stub(status))
        self.assertEqual(
            [("external/mstd", "is checked out at a different commit than the one recorded", True),
             ("external/devops", "is not initialised", False),
             ("external/gt", "has merge conflicts", False)], problems)

    def test_clean_submodules_and_garbage_give_nothing(self):
        self.assertEqual([], snap.submodule_problems("/r", self.run_stub(" aaa external/ok (v1)\n\n+x\n")))
        self.assertEqual([], snap.submodule_problems("/r", self.run_stub("")))


class DetailCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        store.save_snapshot(self.root, make_snapshot("20261006T100000Z"))
        store.save_snapshot(self.root, make_snapshot("20261007T100000Z"))
        store.save_detail(self.root, sample_detail())

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = pqbt.main(["--data-dir", str(self.root), *args])
        return code, buffer.getvalue()

    def test_detail_prints_the_report_of_a_snapshot(self):
        code, text = self.run_cli("detail", "20261006", "--top", "3")
        self.assertEqual(0, code)
        self.assertIn("== Build at a glance ==", text)
        self.assertIn("snapshot 20261006T100000Z", text)

    def test_a_snapshot_without_detail_is_explained(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("detail")   # the latest has no detail
        self.assertIn("take a new snapshot", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("detail", "nope")
        self.assertIn("no such snapshot", str(caught.exception))

    def test_a_used_stale_submodule_stops_the_snapshot_before_building(self):
        problems = [("external/mstd", "is checked out at a different commit than the one recorded", True)]
        with mock.patch.object(snap, "submodule_problems", return_value=problems), \
                mock.patch.object(snap, "take_snapshot", side_effect=AssertionError("must not build")), \
                mock.patch.object(os, "getloadavg", return_value=(0.1, 0.1, 0.1)), \
                contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as caught:
            pqbt.main(["--data-dir", str(self.root), "snapshot", "--build-dir", str(self.root / "b")])
        self.assertIn("git submodule update", str(caught.exception))
        self.assertIn("the build uses it", errors.getvalue())

    def test_force_and_unused_submodules_only_warn(self):
        for used, extra in ((True, ["--force"]), (False, [])):
            problems = [("external/x", "is not initialised", used)]
            with mock.patch.object(snap, "submodule_problems", return_value=problems), \
                    mock.patch.object(snap, "take_snapshot", return_value=(make_snapshot("20261008T100000Z"), sample_detail())), \
                    mock.patch.object(os, "getloadavg", return_value=(0.1, 0.1, 0.1)), \
                    contextlib.redirect_stderr(io.StringIO()) as errors, contextlib.redirect_stdout(io.StringIO()):
                pqbt.main(["--data-dir", str(self.root), "snapshot", "--build-dir", str(self.root / "b"), *extra])
            self.assertIn("warning: submodule external/x", errors.getvalue())
        self.assertTrue((self.root / "snapshots" / "20261008T100000Z.json").exists())
        self.assertTrue((self.root / "details" / "20261006T100000Z.json").exists())



def with_steps(*items):
    """A minimal detail for render_changes: items are (source, seconds, repository files)."""
    return {"source": "cold", "steps": [{"o": f"{src}.o", "k": "compile", "s": seconds, "src": src, "p": files} for src, seconds, files in items],
            "headers": []}


class BurdenGroupingTests(unittest.TestCase):
    CHURN = {"include/x.hpp": 2, "include/x.tpp": 2, "include/y.hpp": 2, "include/z.hpp": 2}

    def burden(self, churn):
        deps = {OBJ_A: ["../include/x.hpp", "../include/x.tpp", "../include/y.hpp"], OBJ_B: ["../include/x.hpp", "../include/x.tpp", "../include/z.hpp"]}
        built = detail.make_detail("i", "f", 4, "cold", [entry(0, 4000, OBJ_A), entry(0, 3000, OBJ_B)], deps, {}, ROOT, BUILD, churn, "dev")
        return detail.render(built, sample_snapshot(), 12).split("By burden:")[1].split("By rebuild cost")[0]

    def test_headers_with_the_same_includers_and_changes_are_one_row(self):
        section = self.burden(self.CHURN)
        self.assertIn("include/x.hpp  +1 with the same includers: x.tpp", section)
        self.assertEqual(1, section.count("x.tpp"))
        self.assertIn("include/y.hpp", section)
        self.assertIn("include/z.hpp", section)   # different includers: its own row

    def test_the_same_includers_but_a_different_number_of_changes_stay_apart(self):
        section = self.burden(dict(self.CHURN, **{"include/x.tpp": 5}))
        self.assertEqual(2, len([line for line in section.splitlines() if "x.hpp" in line or "x.tpp" in line]))
        self.assertNotIn("with the same includers: x.tpp", section)


class ChangesBetweenDetailsTests(unittest.TestCase):
    def test_exact_include_changes_come_first_and_sort_by_size(self):
        before = with_steps(("src/a.cpp", 4.0, 30), ("src/b.cpp", 3.0, 20), ("src/c.cpp", 2.0, 10))
        after = with_steps(("src/a.cpp", 4.0, 29), ("src/b.cpp", 3.0, 5), ("src/c.cpp", 2.0, 10))
        text = detail.render_changes(before, after)
        rows = [line.split() for line in text.split("\n\n")[0].splitlines()[3:]]
        self.assertEqual([["20", "5", "-15", "src/b.cpp"], ["30", "29", "-1", "src/a.cpp"]], rows)   # c.cpp is unchanged, so absent

    def test_added_and_removed_translation_units(self):
        text = detail.render_changes(with_steps(("src/a.cpp", 1.0, 5), ("src/b.cpp", 1.0, 5)), with_steps(("src/a.cpp", 1.0, 5), ("src/c.cpp", 1.0, 5)))
        self.assertIn("1 translation units added, 1 removed", text)

    def test_header_fan_in_changes_are_exact_and_cover_headers_outside_the_top_list(self):
        before = {"source": "cold", "steps": [], "headers": [
            {"f": "include/core.hpp", "fan_in": 100, "cpu_s": 600.0}, {"f": "include/same.hpp", "fan_in": 5, "cpu_s": 9.0}],
            "fan_in": {"include/core.hpp": 100, "include/same.hpp": 5, "include/small.hpp": 4, "include/old.hpp": 3}}
        after = {"source": "cold", "steps": [], "headers": [
            {"f": "include/core.hpp", "fan_in": 60, "cpu_s": 360.0}, {"f": "include/same.hpp", "fan_in": 5, "cpu_s": 99.0}],
            "fan_in": {"include/core.hpp": 60, "include/same.hpp": 5, "include/small.hpp": 1, "include/new.hpp": 2}}
        text = detail.render_changes(before, after)
        section = text.split("Headers whose fan-in")[1].split("Compile CPU per module")[0]
        rows = [line.split() for line in section.splitlines() if " -> " in line and "hpp" in line]
        self.assertEqual([["100", "->", "60", "10m", "00s", "->", "6m", "00s", "include/core.hpp"], ["4", "->", "1", "-", "include/small.hpp"]], rows)
        self.assertNotIn("same.hpp", section)   # an unchanged fan-in with a different time is only timing noise
        self.assertIn("1 headers added to the build, 1 no longer included by anything", text)   # new.hpp and old.hpp

    def test_without_the_full_fan_in_map_the_top_list_is_used(self):
        before = {"source": "cold", "steps": [], "headers": [{"f": "include/a.hpp", "fan_in": 9, "cpu_s": 3.0}]}
        after = {"source": "cold", "steps": [], "headers": [{"f": "include/a.hpp", "fan_in": 4, "cpu_s": 1.0}]}
        self.assertIn("9 -> 4", detail.render_changes(before, after))

    def test_a_detail_without_the_full_map_is_compared_by_the_top_lists_on_both_sides(self):
        top = [{"f": "include/a.hpp", "fan_in": 9, "cpu_s": 3.0}]
        old = {"source": "cold", "steps": [], "headers": top}
        new = {"source": "cold", "steps": [], "headers": top, "fan_in": {"include/a.hpp": 9, "include/b.hpp": 1, "include/c.hpp": 1}}
        text = detail.render_changes(old, new)
        self.assertNotIn("added to the build", text)   # b.hpp and c.hpp are only unknown on the old side
        self.assertIn("none", text.split("Headers whose fan-in")[1].split("Compile CPU")[0])

    def test_many_changed_headers_are_counted(self):
        before = {"source": "cold", "steps": [], "headers": [], "fan_in": {f"include/h{i}.hpp": 10 for i in range(5)}}
        after = {"source": "cold", "steps": [], "headers": [], "fan_in": {f"include/h{i}.hpp": 9 for i in range(5)}}
        self.assertIn("... and 3 more", detail.render_changes(before, after, top=2).split("Headers whose fan-in")[1])

    def test_modules_are_only_reported_beyond_the_noise(self):
        before = with_steps(("src/a/a.cpp", 100.0, 1), ("src/e/e.cpp", 100.0, 1), ("src/b/b.cpp", 100.0, 1), ("src/c/c.cpp", 50.0, 1), ("src/d/d.cpp", 10.0, 1))
        after = with_steps(("src/a/a.cpp", 100.0, 1), ("src/e/e.cpp", 100.0, 1), ("src/b/b.cpp", 130.0, 1), ("src/c/c.cpp", 20.0, 1), ("src/d/d.cpp", 6.0, 1))
        text = detail.render_changes(before, after)
        section = text.split("Compile CPU per module")[1]
        self.assertIn("whole build 6m 00s -> 5m 56s, the typical module ran x1.00", section)
        self.assertNotIn("src/a", section)   # unchanged
        self.assertNotIn("src/d", section)   # -4 s: under 5 s
        rows = [line.split() for line in section.splitlines()[4:]]
        self.assertEqual(["src/b", "src/c"], [row[-1] for row in rows])   # both moved by 30 s: ties by name
        self.assertIn("+30%", rows[0])
        self.assertIn("-60%", rows[1])
        self.assertIn("+30s", " ".join(rows[0]))

    def test_a_change_must_exceed_both_the_seconds_and_the_percentage(self):
        steady = [(f"src/m{i}/f.cpp", 200.0, 1) for i in range(5)]
        before = with_steps(*steady, ("src/big/b.cpp", 200.0, 1), ("src/small/s.cpp", 10.0, 1))
        after = with_steps(*steady, ("src/big/b.cpp", 214.0, 1), ("src/small/s.cpp", 14.0, 1))   # +7% of 200 s; +4 s of 10 s
        self.assertIn("no module changed beyond the noise", detail.render_changes(before, after))
        after = with_steps(*steady, ("src/big/b.cpp", 218.0, 1), ("src/small/s.cpp", 16.0, 1))    # +9% (> 8%); +6 s (> 5 s)
        section = detail.render_changes(before, after).split("Compile CPU per module")[1]
        self.assertIn("src/big", section)
        self.assertIn("src/small", section)

    def test_a_module_holding_most_of_the_cpu_cannot_be_told_from_the_drift(self):
        # documented limit: the drift is the ratio at which half of the CPU lies, so a module with more than half decides it
        before = with_steps(("src/big/b.cpp", 1000.0, 1), ("src/small/s.cpp", 100.0, 1))
        after = with_steps(("src/big/b.cpp", 1300.0, 1), ("src/small/s.cpp", 100.0, 1))
        text = detail.render_changes(before, after)
        self.assertIn("the typical module ran x1.30", text)
        self.assertIn("whole build 18m 20s -> 23m 20s", text)   # the whole-build line still shows it

    def test_a_slower_machine_is_not_a_change_of_every_module(self):
        before = with_steps(("src/a/a.cpp", 100.0, 1), ("src/b/b.cpp", 100.0, 1), ("src/c/c.cpp", 100.0, 1))
        after = with_steps(("src/a/a.cpp", 160.0, 1), ("src/b/b.cpp", 160.0, 1), ("src/c/c.cpp", 160.0, 1))
        text = detail.render_changes(before, after)
        self.assertIn("the typical module ran x1.60", text)
        self.assertIn("no module changed beyond the noise", text)

    def test_one_module_changing_is_not_hidden_by_the_drift_estimate(self):
        before = with_steps(("src/a/a.cpp", 100.0, 1), ("src/b/b.cpp", 100.0, 1), ("src/c/c.cpp", 100.0, 1))
        after = with_steps(("src/a/a.cpp", 100.0, 1), ("src/b/b.cpp", 100.0, 1), ("src/c/c.cpp", 200.0, 1))   # only c changed
        section = detail.render_changes(before, after).split("Compile CPU per module")[1]
        self.assertIn("the typical module ran x1.00", section)
        rows = [line.split() for line in section.splitlines()[4:]]
        self.assertEqual(["src/c"], [row[-1] for row in rows])   # a and b are not reported as "faster than expected"

    def test_the_drift_is_the_cpu_weighted_median_ratio(self):
        before = {"big": (1, 100.0), "mid": (1, 30.0), "small": (1, 20.0)}
        after = {"big": (1, 110.0), "mid": (1, 60.0), "small": (1, 5.0)}
        self.assertAlmostEqual(1.1, detail.drift(before, after))   # the big module holds the middle of the CPU
        self.assertEqual(1.0, detail.drift({}, {}))

    def test_nothing_changed_says_so(self):
        same = with_steps(("src/a.cpp", 4.0, 3))
        text = detail.render_changes(same, same)
        self.assertEqual(3, text.count("none") + text.count("no module changed"))

    def test_more_changes_than_rows_are_counted(self):
        before = with_steps(*[(f"src/f{i}.cpp", 1.0, 10) for i in range(5)])
        after = with_steps(*[(f"src/f{i}.cpp", 1.0, 9) for i in range(5)])
        self.assertIn("... and 3 more", detail.render_changes(before, after, top=2))


class CompareDetailCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for ident in ("20261001T100000Z", "20261002T100000Z"):
            store.save_snapshot(self.root, compared_snapshot(ident, {"touch_leaf": scenario(1.0, 6, file="a.cpp")}))

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, ident, source="cold"):
        built = sample_detail(source=source)
        built["id"] = ident
        store.save_detail(self.root, built)

    def compare(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            pqbt.main(["--data-dir", str(self.root), "compare"])
        return buffer.getvalue()

    def test_the_detail_section_needs_two_cold_details(self):
        self.assertNotIn("What changed in the detail", self.compare())
        self.save("20261001T100000Z")
        self.assertNotIn("What changed in the detail", self.compare())   # only one of them has a detail
        self.save("20261002T100000Z")
        self.assertIn("== What changed in the detail ==", self.compare())

    def test_times_from_the_log_are_not_compared(self):
        self.save("20261001T100000Z")
        self.save("20261002T100000Z", source="log")
        self.assertNotIn("What changed in the detail", self.compare())



# --- clang -ftime-trace -----------------------------------------------------------------------------------------------

def x_event(name, ts_s, dur_s, detail=None):
    event = {"ph": "X", "name": name, "ts": round(ts_s * 1e6), "dur": round(dur_s * 1e6), "pid": 1, "tid": 1}
    if detail:
        event["args"] = {"detail": detail}
    return event


def source_events(path, start_s, end_s):
    """An include as clang writes it: an async begin and end event next to each other."""
    common = {"name": "Source", "cat": "Source", "id": 0, "pid": 1, "tid": 1}
    return [dict(common, ph="b", ts=round(start_s * 1e6), args={"detail": path}), dict(common, ph="e", ts=round(end_s * 1e6))]


def trace_a(root="/src"):
    """10 s: frontend 6 s, backend 4 s; a.hpp (3 s) includes b.hpp (1 s); the vector header 0.4 s; two nested instantiations."""
    return (
        [x_event("ExecuteCompiler", 0, 10), x_event("Frontend", 0, 6), x_event("Backend", 6, 4)]
        + source_events(f"{root}/include/b.hpp", 2.0, 3.0) + source_events(f"{root}/include/a.hpp", 1.0, 4.0)
        + source_events("/usr/lib/gcc/x86_64-linux-gnu/14/../../../../include/c++/14/vector", 0.5, 0.9)
        + [x_event("InstantiateClass", 5.0, 1.0, "std::vector<int>"), x_event("InstantiateFunction", 5.2, 0.3, "std::vector<int>::push_back")]
    )


def trace_b(root="/src"):
    """4 s: frontend 3 s, backend 1 s; a.hpp (1 s) and one instantiation of vector<int> (0.5 s)."""
    return ([x_event("ExecuteCompiler", 0, 4), x_event("Frontend", 0, 3), x_event("Backend", 3, 1)]
            + source_events(f"{root}/include/a.hpp", 0.5, 1.5) + [x_event("InstantiateClass", 2.0, 0.5, "std::vector<int>")])


OBJ_T = "tests/CMakeFiles/t.dir/t.cpp.o"


def write_trace(build_dir, obj, events):
    path = Path(build_dir) / (obj[:-2] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


class TraceParsingTests(unittest.TestCase):
    def test_async_begin_and_end_events_are_paired_and_garbage_is_skipped(self):
        raw = source_events("/x/a.hpp", 1.0, 2.5) + [{"ph": "e", "name": "Source", "ts": 9, "id": 0, "cat": "Source", "pid": 1, "tid": 1},
                                                     {"ph": "X", "name": "Frontend", "ts": 0, "dur": "bad"}, "junk", {"ts": 1},
                                                     {"ph": "b", "name": "Source", "ts": 5, "id": 0, "cat": "Source", "pid": 1, "tid": 1},
                                                     x_event("Backend", 3, 1)]
        events = trace.complete_events(raw)
        self.assertEqual(["Source", "Backend"], [e["name"] for e in events])
        self.assertEqual((1_000_000, 1_500_000), (events[0]["ts"], events[0]["dur"]))
        self.assertEqual("/x/a.hpp", trace.detail_of(events[0]))

    def test_an_end_that_precedes_its_begin_is_not_an_event(self):
        raw = [dict(source_events("/x/a.hpp", 5.0, 6.0)[0], ts=10_000_000), dict(source_events("/x/a.hpp", 5.0, 6.0)[1], ts=5_000_000)]
        self.assertEqual([], trace.complete_events(raw))

    def test_self_time_removes_the_nested_events(self):
        events = [x_event("S", 0, 10), x_event("S", 2, 3), x_event("S", 3, 1), x_event("S", 6, 2), x_event("S", 20, 1)]
        selfs = sorted((e["ts"] / 1e6, round(s / 1e6, 3)) for e, s in trace.with_self_times(events))
        self.assertEqual([(0.0, 5.0), (2.0, 2.0), (3.0, 1.0), (6.0, 2.0), (20.0, 1.0)], selfs)   # 10 - 3 - 2; 3 - 1

    def test_nesting_is_per_thread(self):
        a, b = x_event("S", 0, 10), dict(x_event("S", 2, 3), tid=2)
        self.assertEqual([10.0, 3.0], sorted(s / 1e6 for _, s in trace.with_self_times([a, b]))[::-1])

    def test_paths_are_made_relative_or_labelled(self):
        expected = {
            "/src/include/a.hpp": "include/a.hpp", "/src/./include/../include/a.hpp": "include/a.hpp",
            "/usr/lib/gcc/x86_64-linux-gnu/14/../../../../include/c++/14/vector": "<std>/vector",
            "/usr/include/c++/15/bits/stl_vector.h": "<std>/bits/stl_vector.h", "/usr/include/stdio.h": "<system>/stdio.h",
            "/usr/lib/llvm-20/lib/clang/20/include/stddef.h": "<clang>/stddef.h",
            "/home/me/.local/share/pq-build-times/deps/eigen-src/Eigen/Dense": "<eigen>/Eigen/Dense", "/opt/x/y.h": "/opt/x/y.h",
        }
        for path, label in expected.items():
            self.assertEqual(label, trace.clean_path(path, "/src"), path)
        self.assertEqual("/srcother/x.h", trace.clean_path("/srcother/x.h", "/src"))   # a sibling directory is not the repository

    def test_one_file_is_analysed_into_phases_headers_and_templates(self):
        figures = trace.analyse(trace.complete_events(trace_a()), "/src")
        self.assertEqual((10.0, 6.0, 4.0), (figures["total_s"], figures["frontend_s"], figures["backend_s"]))
        headers = {name: (round(own, 3), round(incl, 3)) for name, own, incl in figures["headers"]}
        self.assertEqual({"include/a.hpp": (2.0, 3.0), "include/b.hpp": (1.0, 1.0), "<std>/vector": (0.4, 0.4)}, headers)
        templates = {name: (round(own, 3), round(incl, 3)) for name, own, incl in figures["templates"]}
        self.assertEqual({"std::vector<int>": (0.7, 1.0), "std::vector<int>::push_back": (0.3, 0.3)}, templates)

    def test_without_an_executecompiler_event_the_total_is_frontend_plus_backend(self):
        events = trace.complete_events([x_event("Frontend", 0, 3), x_event("Backend", 3, 1)])
        self.assertEqual(4.0, trace.analyse(events, "/src")["total_s"])

    def test_long_names_and_control_characters_are_cleaned(self):
        text = trace.clip("a\nb\x00" + "x" * 500)
        self.assertEqual(trace.MAX_NAME_CHARS, len(text))
        self.assertNotIn("\n", text)
        self.assertTrue(text.endswith("…"))

    def test_trace_paths(self):
        self.assertEqual("src/CMakeFiles/lib.dir/a.cpp.json", trace.trace_path(OBJ_A))
        self.assertIsNone(trace.trace_path("src/liblib.so"))


class TraceCollectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.build = Path(self._tmp.name) / "build"
        write_trace(self.build, OBJ_A, trace_a())
        write_trace(self.build, OBJ_T, trace_b())

    def tearDown(self):
        self._tmp.cleanup()

    def collect(self, objects=(OBJ_A, OBJ_T)):
        logged = []
        result = trace.collect(self.build, "/src", list(objects), ninja.source_of_object, log=logged.append)
        return result, logged

    def test_the_files_are_summarised_with_their_phases(self):
        result, _ = self.collect()
        self.assertEqual([("src/a.cpp", 10.0, 6.0, 4.0, 3, 2), ("tests/t.cpp", 4.0, 3.0, 1.0, 1, 1)],
                         [(f["src"], f["total_s"], f["frontend_s"], f["backend_s"], f["inclusions"], f["instantiations"]) for f in result["files"]])

    def test_headers_and_templates_are_summed_over_the_files(self):
        result, _ = self.collect()
        headers = {h["f"]: (h["self_s"], h["incl_s"], h["n"], h["tus"]) for h in result["headers"]}
        self.assertEqual({"include/a.hpp": (3.0, 4.0, 2, 2), "include/b.hpp": (1.0, 1.0, 1, 1), "<std>/vector": (0.4, 0.4, 1, 1)}, headers)
        self.assertEqual(["include/a.hpp", "include/b.hpp", "<std>/vector"], [h["f"] for h in result["headers"]])   # by own time
        templates = {t["f"]: (t["self_s"], t["incl_s"], t["n"], t["tus"]) for t in result["templates"]}
        self.assertEqual({"std::vector<int>": (1.2, 1.5, 2, 2), "std::vector<int>::push_back": (0.3, 0.3, 1, 1)}, templates)

    def test_the_totals(self):
        totals = self.collect()[0]["totals"]
        self.assertEqual((14.0, 9.0, 5.0, 4.4, 1.5), (totals["total_s"], totals["frontend_s"], totals["backend_s"],
                                                      totals["header_self_s"], totals["template_self_s"]))
        self.assertEqual((4, 3, 3, 2), (totals["inclusions"], totals["instantiations"], totals["distinct_headers"], totals["distinct_templates"]))

    def test_a_header_included_twice_in_one_file_counts_one_file(self):
        events = trace_b() + source_events("/src/include/a.hpp", 2.6, 2.7)
        write_trace(self.build, OBJ_T, events)
        result = self.collect()[0]
        entry = next(h for h in result["headers"] if h["f"] == "include/a.hpp")
        self.assertEqual((3, 2), (entry["n"], entry["tus"]))   # 3 inclusions in 2 files
        self.assertEqual(2, next(f for f in result["files"] if f["src"] == "tests/t.cpp")["inclusions"])   # both events of that file

    def test_missing_and_unreadable_traces_are_counted_not_fatal(self):
        (self.build / "src/CMakeFiles/lib.dir/broken.cpp.json").write_text("{not json")
        result, logged = self.collect([OBJ_A, OBJ_T, "src/CMakeFiles/lib.dir/gone.cpp.o", "src/CMakeFiles/lib.dir/broken.cpp.o", "src/liblib.so"])
        self.assertEqual((2, 2, 1), (len(result["files"]), result["missing"], result["unreadable"]))
        self.assertTrue(any("skipping trace" in line for line in logged))

    def test_no_traces_give_none(self):
        self.assertIsNone(self.collect(["src/CMakeFiles/lib.dir/gone.cpp.o"])[0])
        self.assertIsNone(self.collect([])[0])

    def test_only_the_top_entries_are_kept(self):
        many = [x_event("ExecuteCompiler", 0, 1)] + [e for i in range(trace.TOP_HEADERS + 20) for e in source_events(f"/src/include/h{i}.hpp", i, i + 0.5)]
        write_trace(self.build, OBJ_A, many)
        result = self.collect([OBJ_A])[0]
        self.assertEqual(trace.TOP_HEADERS, len(result["headers"]))
        self.assertEqual(trace.TOP_HEADERS + 20, result["totals"]["distinct_headers"])   # the totals cover all


def sample_trace(**changes):
    result = {
        "files": [{"src": "src/a.cpp", "total_s": 6.5, "frontend_s": 6.0, "backend_s": 0.5, "inclusions": 3, "instantiations": 2},
                  {"src": "tests/t.cpp", "total_s": 7.5, "frontend_s": 3.0, "backend_s": 4.5, "inclusions": 1, "instantiations": 1}],
        "headers": [{"f": "include/a.hpp", "self_s": 3.0, "incl_s": 4.0, "n": 3, "tus": 2},
                    {"f": "<std>/vector", "self_s": 0.4, "incl_s": 0.4, "n": 1, "tus": 1},
                    {"f": "external/mstd/include/x.hpp", "self_s": 0.2, "incl_s": 0.2, "n": 1, "tus": 1}],
        "templates": [{"f": "std::vector<int>", "self_s": 1.2, "incl_s": 1.5, "n": 2, "tus": 2}],
        "missing": 0, "unreadable": 0,
        "totals": {"total_s": 14.0, "frontend_s": 9.0, "backend_s": 5.0, "header_self_s": 3.6, "template_self_s": 1.2,
                   "inclusions": 5, "instantiations": 2, "distinct_headers": 3, "distinct_templates": 1},
    }
    result.update(changes)
    return result


class TraceReportTests(unittest.TestCase):
    def section(self, built=None, top=12):
        return "\n".join(trace_report.render_section(built or {"trace": sample_trace()}, top))

    def test_without_trace_data_a_hint_says_how_to_get_it(self):
        text = self.section({"trace": None})
        self.assertIn("build with clang", text)
        self.assertIn("--compiler clang++-20", text)

    def test_the_phases_and_the_frontend_split(self):
        text = self.section()
        self.assertIn("compiler CPU 14.0s: frontend (parsing, semantic analysis, template instantiation) 9.0s = 64.3%, "
                      "backend (optimisation, code generation) 5.0s = 35.7%", text)
        self.assertIn("time in headers (their own, without what they include) 3.6s = 40.0%, template instantiation (own) 1.2s = 13.3%", text)
        self.assertIn("these overlap", text)
        self.assertIn("5 inclusions of 3 distinct headers, 2 instantiations of 1 distinct templates", text)

    def test_modules_are_ranked_by_total_with_the_backend_share(self):
        rows = [line.split() for line in self.section().split("By module")[1].split("Headers by")[0].splitlines()[3:] if line.strip()]
        self.assertEqual([["1", "3.0s", "4.5s", "60.0%", "tests"], ["1", "6.0s", "0.5s", "7.7%", "src"]], rows)   # tests 7.5 s, src 6.5 s

    def test_modules_beyond_top_are_summed(self):
        text = self.section(top=1)
        self.assertIn("... 1 more modules", text)
        self.assertEqual("6.0s", [line.split() for line in text.splitlines() if "more modules" in line][0][1])   # the smaller module

    def test_headers_show_own_time_per_file_and_changes_only_with_churn(self):
        text = self.section()
        header = text.split("Headers by")[1].split("Template instantiations")[0]
        self.assertNotIn("PRs", header)
        row = next(line.split() for line in header.splitlines() if line.rstrip().endswith("include/a.hpp"))
        self.assertEqual(["3.0s", "33.3%", "4.0s", "2", "1500", "ms", "include/a.hpp"], row)   # 3.0 s over 2 files (3 inclusions)
        with_churn = sample_trace()
        with_churn["headers"][0]["ch"] = 7
        text = self.section({"trace": with_churn})
        self.assertIn("PRs", text)
        self.assertIn("7", next(line for line in text.splitlines() if line.rstrip().endswith("include/a.hpp")).split())

    def test_templates_and_backend_heavy_files(self):
        text = self.section()
        row = next(line.split() for line in text.splitlines() if line.rstrip().endswith("std::vector<int>"))
        self.assertEqual(["1.2s", "13.3%", "1.5s", "2", "2", "std::vector<int>"], row)
        heavy = text.split("takes the most")[1].splitlines()[3:]
        self.assertEqual(["4.5s", "3.0s", "60.0%", "tests/t.cpp"], heavy[0].split())   # by backend time, not by frontend time
        self.assertEqual(["0.5s", "6.0s", "7.7%", "src/a.cpp"], heavy[1].split())

    def test_unreadable_and_missing_traces_are_mentioned(self):
        self.assertIn("(3 files without a trace, 1 unreadable)", self.section({"trace": sample_trace(missing=3, unreadable=1)}))

    def test_the_section_is_part_of_the_detail_report_and_the_hint_otherwise(self):
        built = sample_detail()
        self.assertIn("== Compiler time (clang -ftime-trace) ==\nno clang trace data", detail.render(built, sample_snapshot(), 12))
        built["trace"] = sample_trace()
        text = detail.render(built, sample_snapshot(), 12)
        self.assertIn("== Compiler time (clang -ftime-trace) ==\ncompiler CPU 14.0s", text)
        self.assertLess(text.index("== Slowest compile steps =="), text.index("== Compiler time"))
        self.assertLess(text.index("== Compiler time"), text.index("== Headers: what a change rebuilds =="))


class TraceNoiseTests(unittest.TestCase):
    def test_the_drift_is_the_time_weighted_median_ratio(self):
        self.assertAlmostEqual(1.1, trace_report.drift_of({"a": 10.0, "b": 10.0}, {"a": 11.0, "b": 11.0}))
        self.assertAlmostEqual(1.0, trace_report.drift_of({"big": 100.0, "small": 1.0}, {"big": 100.0, "small": 5.0}))   # the big one decides
        self.assertEqual(1.0, trace_report.drift_of({}, {}))
        self.assertEqual(1.0, trace_report.drift_of({"a": 0.0}, {"a": 5.0}))

    def test_a_big_entry_outweighs_two_small_ones_in_the_drift(self):
        before, after = {"big": 100.0, "s1": 1.0, "s2": 1.0}, {"big": 100.0, "s1": 2.0, "s2": 3.0}
        self.assertEqual(1.0, trace_report.drift_of(before, after))   # an unweighted median of 1, 2 and 3 would be 2

    def test_the_biggest_change_comes_first(self):
        steady = {"x": 50.0, "y": 50.0}
        found, _ = trace_report.movers({**steady, "a": 10.0, "b": 10.0, "c": 10.0}, {**steady, "a": 12.0, "b": 18.0, "c": 6.0}, 1.0, 0.15)
        self.assertEqual(["b", "c", "a"], [name for _, name, *_ in found])   # +8 s, -4 s, +2 s

    def moved(self, before, after, absolute=1.0, relative=0.15):
        found, scale = trace_report.movers(before, after, absolute, relative)
        return [name for _, name, *_ in found], scale

    def test_a_change_must_exceed_the_seconds_and_the_percentage(self):
        steady = {"x": 50.0, "y": 50.0}
        self.assertEqual(["h"], self.moved({**steady, "h": 10.0}, {**steady, "h": 11.6})[0])        # +1.6 s: > 1 s and > 15%
        self.assertEqual([], self.moved({**steady, "h": 10.0}, {**steady, "h": 10.9})[0])           # +0.9 s: under 1 s
        self.assertEqual([], self.moved({**steady, "h": 10.0}, {**steady, "h": 11.4})[0])           # +1.4 s but only 14%
        self.assertEqual([], self.moved({**steady, "big": 100.0}, {**steady, "big": 114.0})[0])     # +14 s but only 14%
        self.assertEqual(["big"], self.moved({**steady, "big": 100.0}, {**steady, "big": 116.0})[0])

    def test_improvements_are_found_too(self):
        steady = {"x": 50.0, "y": 50.0}
        found, _ = trace_report.movers({**steady, "h": 10.0}, {**steady, "h": 5.0}, 1.0, 0.15)
        self.assertEqual(("h", -5.0), (found[0][1], found[0][0]))

    def test_a_slower_machine_is_not_a_change_of_everything(self):
        before = {"a": 10.0, "b": 20.0, "c": 30.0}
        names, scale = self.moved(before, {k: v * 1.3 for k, v in before.items()})
        self.assertEqual(([], 1.3), (names, round(scale, 3)))

    def test_only_entries_present_on_both_sides_are_compared_and_ties_sort_by_name(self):
        steady = {"x": 50.0, "y": 50.0}
        names, _ = self.moved({**steady, "gone": 10.0, "b": 10.0, "a": 10.0}, {**steady, "new": 10.0, "b": 20.0, "a": 20.0})
        self.assertEqual(["a", "b"], names)


class TraceChangesTests(unittest.TestCase):
    def changed(self, **mutations):
        before, after = sample_trace(), sample_trace()
        after = json.loads(json.dumps(after))
        for key, value in mutations.items():
            value(after)
        return "\n".join(trace_report.render_changes(before, after))

    def big(self):
        trace_data = sample_trace()
        trace_data["headers"] += [{"f": f"include/steady{i}.hpp", "self_s": 50.0, "incl_s": 50.0, "n": 1, "tus": 1} for i in range(3)]
        trace_data["templates"] += [{"f": f"steady<{i}>", "self_s": 50.0, "incl_s": 50.0, "n": 1, "tus": 1} for i in range(3)]
        trace_data["files"] += [{"src": f"src/m{i}/f.cpp", "total_s": 100.0, "frontend_s": 60.0, "backend_s": 40.0, "inclusions": 1, "instantiations": 1}
                                for i in range(3)]
        return trace_data

    def test_identical_data_lists_nothing(self):
        text = "\n".join(trace_report.render_changes(self.big(), self.big()))
        self.assertEqual(3, text.count("none beyond the noise"))
        self.assertIn("frontend x1.00, backend x1.00", text)

    def test_the_totals_are_always_shown(self):
        after = self.big()
        after["totals"] = dict(after["totals"], total_s=15.4, frontend_s=9.9, backend_s=5.5)
        text = "\n".join(trace_report.render_changes(self.big(), after))
        self.assertIn("compiler CPU 14.0s -> 15.4s (+10.0%): frontend 9.0s -> 9.9s (+10.0%), backend 5.0s -> 5.5s (+10.0%)", text)

    def test_a_header_template_and_module_phase_that_moved_are_listed(self):
        after = self.big()
        after["headers"][0]["self_s"] = 1.0              # include/a.hpp: 3.0 s -> 1.0 s  (-2.0 s, -67%)
        after["templates"][0]["self_s"] = 3.0            # std::vector<int>: 1.2 s -> 3.0 s
        for item in after["files"][2:]:                  # src/m0: the backend of one module doubles
            item["backend_s"] = 80.0 if item["src"] == "src/m0/f.cpp" else item["backend_s"]
        text = "\n".join(trace_report.render_changes(self.big(), after))
        headers = text.split("Headers whose own parse time moved")[1].split("Templates whose")[0]
        self.assertIn("3.0s -> 1.0s", headers)
        self.assertIn("-67%", headers)
        self.assertIn("include/a.hpp", headers)
        self.assertNotIn("steady", headers)
        templates = text.split("Templates whose own time moved")[1]
        self.assertIn("1.2s -> 3.0s", templates)
        self.assertIn("+150%", templates)
        modules = text.split("Modules whose frontend or backend time moved")[1].split("Headers whose")[0]
        row = next(line.split() for line in modules.splitlines() if line.rstrip().endswith("src/m0"))
        self.assertEqual("backend", row[0])
        self.assertIn("40.0s", row)

    def test_the_calibrated_thresholds_are_pinned(self):
        # measured on three identical clang builds (README); changing them needs a new measurement
        self.assertEqual((1.0, 0.15, 5.0, 0.08), (trace_report.ITEM_NOISE_S, trace_report.ITEM_NOISE_RELATIVE,
                                                  trace_report.PHASE_NOISE_S, trace_report.PHASE_NOISE_RELATIVE))

    def test_the_thresholds_are_applied_at_their_boundaries(self):
        def listed(mutate):
            after = self.big()
            mutate(after)
            return "\n".join(trace_report.render_changes(self.big(), after))

        def header(value):
            return lambda t: t["headers"][0].update(self_s=value)

        self.assertNotIn("include/a.hpp", listed(header(3.9)).split("Headers whose own")[1])   # +0.9 s: under 1 s
        self.assertIn("include/a.hpp", listed(header(4.2)).split("Headers whose own")[1])      # +1.2 s and +40%

        def frontend(value):
            def mutate(t):
                t["files"][2]["frontend_s"] = value
            return mutate

        self.assertNotIn("src/m0", listed(frontend(64.0)).split("Modules whose")[1].split("Headers whose")[0])   # +4 s: under 5 s
        self.assertIn("src/m0", listed(frontend(66.0)).split("Modules whose")[1].split("Headers whose")[0])      # +6 s and +10%

    def test_top_limits_the_module_rows(self):
        def with_steady_modules():   # enough unchanged modules that the drift estimate stays at 1
            data = self.big()
            data["files"] += [{"src": f"src/s{i}/f.cpp", "total_s": 100.0, "frontend_s": 60.0, "backend_s": 40.0, "inclusions": 1, "instantiations": 1}
                              for i in range(3)]
            return data

        before, after = with_steady_modules(), with_steady_modules()
        for item in after["files"]:
            item["backend_s"] += 30.0 if item["src"] == "src/m0/f.cpp" else 20.0 if item["src"] == "src/m1/f.cpp" else 0.0
        text = "\n".join(trace_report.render_changes(before, after, top=1))
        modules = text.split("Modules whose")[1].split("Headers whose")[0]
        rows = [line for line in modules.splitlines() if line.rstrip().endswith(("src/m0", "src/m1", "src/m2"))]
        self.assertEqual(1, len(rows))   # one row (the one that moved most), not two
        self.assertIn("src/m0", modules)
        self.assertNotIn("src/m1", modules)

    def test_more_changes_than_rows_are_counted(self):
        after = self.big()
        before = self.big()
        before["headers"] = [dict(h, f=f"include/h{i}.hpp", self_s=10.0) for i, h in enumerate(before["headers"] * 2)]
        after["headers"] = [dict(h, f=f"include/h{i}.hpp", self_s=40.0 if i < 5 else 10.0) for i, h in enumerate(after["headers"] * 2)]
        text = "\n".join(trace_report.render_changes(before, after, top=2))
        self.assertIn("... and 3 more", text)

    def test_compare_adds_the_section_only_when_both_snapshots_have_trace_data(self):
        a, b = sample_detail(), sample_detail()
        self.assertNotIn("Compiler time changes", detail.render_changes(a, b))
        a["trace"] = self.big()
        self.assertNotIn("Compiler time changes", detail.render_changes(a, b))
        b["trace"] = self.big()
        self.assertIn("== Compiler time changes (clang -ftime-trace) ==", detail.render_changes(a, b))


class TraceSnapshotTests(SnapshotTests):
    def setUp(self):
        super().setUp()
        self.build = self.root / "build-times"
        self.repo = str(self.root.resolve())
        write_trace(self.build, OBJ_A, trace_a(self.repo))
        write_trace(self.build, OBJ_B, trace_b(self.repo))

    def builder(self):
        builder = FakeBuilder()
        builder.source_root = self.repo   # the dependencies are relative to the build directory, so the sources must be under this root
        builder.build_dir = str(self.build)
        return builder

    def test_the_cold_snapshot_stores_the_trace_summary(self):
        self.take(self.builder())
        stored = self.detail["trace"]
        self.assertEqual(["src/a.cpp", "src/b.cpp"], [f["src"] for f in stored["files"]])
        self.assertEqual((14.0, 9.0, 5.0), (stored["totals"]["total_s"], stored["totals"]["frontend_s"], stored["totals"]["backend_s"]))
        self.assertEqual("include/a.hpp", stored["headers"][0]["f"])

    def test_the_traces_are_read_before_the_touch_scenarios_rewrite_them(self):
        builder = self.builder()
        touched = []

        def rewrite():
            touched.append(1)
            write_trace(self.build, OBJ_A, [x_event("ExecuteCompiler", 0, 99), x_event("Frontend", 0, 99)])

        builder.on_touch = rewrite
        self.take(builder)
        self.assertEqual(3, len(touched) // 2)   # the scenarios did run: three touch scenarios, two repetitions each
        self.assertEqual(14.0, self.detail["trace"]["totals"]["total_s"])   # not 99 + 4

    def test_without_the_cold_scenario_the_existing_traces_are_read(self):
        self.take(self.builder(), scenarios=("noop",))
        self.assertEqual(14.0, self.detail["trace"]["totals"]["total_s"])

    def test_headers_of_the_repository_get_their_change_counts(self):
        with mock.patch.object(detail, "git_churn", return_value=({"include/a.hpp": 5, "external/mstd": 2}, "origin/dev")):
            self.take(self.builder())
        by_name = {h["f"]: h for h in self.detail["trace"]["headers"]}
        self.assertEqual(5, by_name["include/a.hpp"]["ch"])
        self.assertNotIn("ch", by_name["<std>/vector"])   # system headers are not ours to change

    def test_a_build_without_traces_has_none(self):
        builder = FakeBuilder()   # its build directory does not exist
        self.take(builder)
        self.assertIsNone(self.detail["trace"])


class TraceChurnTests(unittest.TestCase):
    def test_repository_system_and_submodule_headers(self):
        churn = {"include/a.hpp": 3, "external/mstd": 2}
        built = detail.make_detail("i", "f", 1, "cold", [entry(0, 1000, OBJ_A)], {}, {}, ROOT, BUILD, churn, "dev", 180, sample_trace())
        by_name = {h["f"]: h.get("ch", "none") for h in built["trace"]["headers"]}
        self.assertEqual({"include/a.hpp": 3, "<std>/vector": "none", "external/mstd/include/x.hpp": 2}, by_name)

    def test_without_history_nothing_is_added(self):
        built = detail.make_detail("i", "f", 1, "cold", [entry(0, 1000, OBJ_A)], {}, {}, ROOT, BUILD, None, None, 180, sample_trace())
        self.assertTrue(all("ch" not in h for h in built["trace"]["headers"]))


class CompilerOptionTests(unittest.TestCase):
    def test_the_c_compiler_follows_the_cxx_compiler(self):
        expected = {"clang++-20": "clang-20", "clang++": "clang", "/usr/bin/clang++-19": "/usr/bin/clang-19", "g++-14": "gcc-14",
                    "/opt/gcc/bin/g++": "/opt/gcc/bin/gcc", "c++": "cc", "icpx": None}
        for cxx, c in expected.items():
            self.assertEqual(c, snap.c_compiler_for(cxx), cxx)

    def test_clang_gets_time_trace_gcc_does_not(self):
        self.assertEqual(["-DCMAKE_CXX_COMPILER=clang++-20", "-DCMAKE_C_COMPILER=clang-20", "-DCMAKE_CXX_FLAGS=-ftime-trace"],
                         snap.compiler_arguments("clang++-20", []))
        self.assertEqual(["-DCMAKE_CXX_COMPILER=g++-14", "-DCMAKE_C_COMPILER=gcc-14"], snap.compiler_arguments("g++-14", []))

    def test_own_cxx_flags_are_not_overridden(self):
        self.assertNotIn("-DCMAKE_CXX_FLAGS=-ftime-trace", snap.compiler_arguments("clang++-20", ["-DCMAKE_CXX_FLAGS=-O1"]))

    def test_an_unknown_compiler_is_refused_with_a_hint(self):
        with self.assertRaises(snap.SnapshotError) as caught:
            snap.compiler_arguments("icpx", [])
        self.assertIn("C compiler", str(caught.exception))


class CompilerCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def snapshot(self, *args):
        seen = {}

        def fake_take(builder, *a, **k):
            seen["args"] = builder.cmake_args
            return make_snapshot("20261008T100000Z"), sample_detail()

        with mock.patch.object(snap, "take_snapshot", side_effect=fake_take), mock.patch.object(snap, "submodule_problems", return_value=[]), \
                mock.patch.object(os, "getloadavg", return_value=(0.1, 0.1, 0.1)), contextlib.redirect_stdout(io.StringIO()):
            pqbt.main(["--data-dir", str(self.root), "snapshot", "--build-dir", str(self.root / "b"), *args])
        return seen["args"]

    def test_compiler_selects_both_compilers_and_the_trace_flag(self):
        args = self.snapshot("--compiler", "clang++-20")
        for expected in ("-DCMAKE_CXX_COMPILER=clang++-20", "-DCMAKE_C_COMPILER=clang-20", "-DCMAKE_CXX_FLAGS=-ftime-trace"):
            self.assertIn(expected, args)

    def test_a_cmake_arg_comes_after_and_so_wins(self):
        args = self.snapshot("--compiler", "clang++-20", "--cmake-arg=-DCMAKE_C_COMPILER=/usr/bin/cc")
        self.assertLess(args.index("-DCMAKE_C_COMPILER=clang-20"), args.index("-DCMAKE_C_COMPILER=/usr/bin/cc"))

    def test_gcc_gets_no_trace_flag_and_unknown_compilers_stop_the_run(self):
        self.assertNotIn("-DCMAKE_CXX_FLAGS=-ftime-trace", self.snapshot("--compiler", "g++-14"))
        with self.assertRaises(SystemExit) as caught:
            self.snapshot("--compiler", "icpx")
        self.assertIn("C compiler", str(caught.exception))

    def test_without_the_option_nothing_is_added(self):
        self.assertFalse([a for a in self.snapshot() if a.startswith(("-DCMAKE_CXX_COMPILER=", "-DCMAKE_C_COMPILER=", "-DCMAKE_CXX_FLAGS="))])



class ForeignCpuTests(unittest.TestCase):
    def test_the_machine_busy_time_is_read_from_proc_stat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stat"
            # user nice system idle iowait irq softirq steal guest guest_nice: busy = 1000 + 50 + 200 + 10 + 20 + 5 = 1285 ticks
            path.write_text("cpu  1000 50 200 99999 777 10 20 5 300 0\ncpu0 1 1 1 1 1 1 1 1 1 1\n")
            self.assertAlmostEqual(1285 / os.sysconf("SC_CLK_TCK"), snap.machine_busy_seconds(str(path)))
            path.write_text("cpu  1000 50 200 99999\n")   # an old kernel without the later columns
            self.assertAlmostEqual((1000 + 50 + 200) / os.sysconf("SC_CLK_TCK"), snap.machine_busy_seconds(str(path)))

    def test_an_unreadable_or_foreign_stat_file_gives_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stat"
            for text in ("", "intr 1 2 3\n", "cpu  a b c d e f g h\n"):
                path.write_text(text)
                self.assertIsNone(snap.machine_busy_seconds(str(path)), repr(text))
        self.assertIsNone(snap.machine_busy_seconds("/nonexistent/stat"))

    def test_children_cpu_is_a_non_negative_number(self):
        self.assertGreaterEqual(snap.children_cpu_seconds(), 0.0)

    def test_foreign_cpu_is_the_machine_time_that_our_children_did_not_use(self):
        self.assertEqual(40.0, snap.foreign_cpu((100.0, 10.0), (190.0, 60.0)))   # 90 s busy, 50 s ours
        self.assertEqual(0.0, snap.foreign_cpu((100.0, 10.0), (140.0, 60.0)))    # 40 s busy but 50 s ours (rounding of ticks): never negative
        self.assertIsNone(snap.foreign_cpu((None, 10.0), (150.0, 60.0)))
        self.assertIsNone(snap.foreign_cpu((100.0, 10.0), (None, 60.0)))

    def builder(self, probes, directory):
        build_dir = Path(directory) / "bt"
        builder = snap.Builder(directory, str(build_dir), str(Path(directory) / "deps"), "Release", [], "all", 4,
                               run=lambda command, cwd=None: (0, "ok"), clock=iter(range(0, 100, 5)).__next__, probe=iter(probes).__next__)
        builder.claim_build_dir()
        return builder

    def test_a_build_records_the_foreign_cpu_between_its_two_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.builder([(100.0, 10.0), (190.0, 60.0)], directory).build()
        self.assertEqual(40.0, result["foreign_cpu_s"])

    def test_without_a_probe_value_nothing_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.builder([(None, 0.0), (None, 0.0)], directory).build()
        self.assertNotIn("foreign_cpu_s", result)

    def test_the_aggregate_keeps_the_worst_repetition(self):
        run = lambda foreign: {"wall_s": 1.0, "steps": 1, "cpu_s": 1.0, "link_s": 0.0, "tail_s": 0.0, **({} if foreign is None else {"foreign_cpu_s": foreign})}
        self.assertEqual(30.0, snap.aggregate([run(2.0), run(30.0), run(5.0)])["foreign_cpu_s"])
        self.assertEqual(5.0, snap.aggregate([run(None), run(5.0)])["foreign_cpu_s"])
        self.assertNotIn("foreign_cpu_s", snap.aggregate([run(None), run(None)]))


def with_foreign(snapshot, scenario, foreign, cpu=1200.0):
    snapshot["scenarios"][scenario] = dict(snapshot["scenarios"].get(scenario, {}), foreign_cpu_s=foreign, cpu_s=cpu)
    return snapshot


class InterferenceTests(unittest.TestCase):
    def test_a_build_is_flagged_above_five_percent_and_ten_seconds(self):
        base = make_snapshot("20261001T100000Z")
        self.assertEqual([], compare.interference(with_foreign(dict(base, scenarios={}), "cold", 25.0)))          # the idle baseline: 2%
        self.assertEqual([], compare.interference(with_foreign(dict(base, scenarios={}), "cold", 59.0)))          # under 5% of 1200 s
        self.assertEqual([("cold", 61.0, 1200.0)], compare.interference(with_foreign(dict(base, scenarios={}), "cold", 61.0)))
        self.assertEqual([], compare.interference(with_foreign(dict(base, scenarios={}), "touch_leaf", 9.0, cpu=20.0)))      # under 10 s
        self.assertEqual([("touch_leaf", 10.0, 20.0)], compare.interference(with_foreign(dict(base, scenarios={}), "touch_leaf", 10.0, cpu=20.0)))

    def test_old_snapshots_without_the_figure_are_not_flagged(self):
        self.assertEqual([], compare.interference(make_snapshot("20261001T100000Z")))

    def test_the_warning_names_the_snapshot_the_scenario_and_the_share(self):
        snapshot = with_foreign(dict(make_snapshot("20261001T100000Z"), scenarios={}), "cold", 116.0, cpu=1300.0)
        self.assertEqual(["warning: other processes used about 116 s of CPU during the cold build of 20261001T100000Z (9% of the build's CPU); "
                          "its timings are inflated, repeat it on an idle machine"], compare.interference_warnings(snapshot))

    def test_compare_and_detail_show_the_warning(self):
        a = compared_snapshot("20261001T100000Z", {"cold": scenario(40.0, 600, runs=1)})
        b = with_foreign(compared_snapshot("20261002T100000Z", {"cold": scenario(43.0, 600, runs=1)}), "cold", 116.0, cpu=1300.0)
        text = compare.render(compare.compare(a, b))
        self.assertIn("other processes used about 116 s of CPU during the cold build of 20261002T100000Z", text)
        self.assertNotIn("20261001T100000Z (", text.split("warning:")[1])   # only the disturbed one is named
        self.assertIn("other processes used about 116 s", detail.render(sample_detail(), b, 12))
        self.assertNotIn("other processes used", detail.render(sample_detail(), a, 12))


if __name__ == "__main__":
    unittest.main()
