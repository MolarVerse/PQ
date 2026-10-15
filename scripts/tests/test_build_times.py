import contextlib
import io
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

        builder = self.builder(target, run=run, clock=iter([100.0, 107.5]).__next__)
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


def sample_detail(touches=None, churn=DETAIL_CHURN, source="cold", entries=DETAIL_ENTRIES):
    return detail.make_detail("20261006T100000Z", "abc123abc123", 8, source, entries, DETAIL_DEPS, touches or {}, ROOT, BUILD,
                              churn, "origin/dev" if churn is not None else None, 180)


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


if __name__ == "__main__":
    unittest.main()
