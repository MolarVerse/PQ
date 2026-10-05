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
            "src/a.o: #deps 3, deps mtime 1 (VALID)\n    ../src/a.cpp\n    ../include/core.hpp\n    ../include/mid.hpp\n\n"
            "src/b.o: #deps 3, deps mtime 1 (VALID)\n    ../src/b.cpp\n    ../include/core.hpp\n    ../include/mid.hpp\n\n"
        )
        self.built = built
        self.clock = 0

    def _result(self, steps, wall):
        return {"wall_s": wall, "steps": steps, "cpu_s": wall * 4, "link_s": 1.0, "tail_s": 0.5}

    def cold(self):
        self.calls.append("cold")
        result = self._result(100, 100.0)
        result["configure_s"] = 5.0
        return result

    def ensure_built(self):
        self.calls.append("ensure_built")
        if not self.built:
            raise snap.SnapshotError("nothing to measure yet")

    def build(self):
        self.calls.append("build")
        return self._result(3, 2.0)

    def deps_text(self):
        return self.deps

    def touch(self, relative):
        self.calls.append(f"touch {relative}")

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
        return snap.take_snapshot(
            builder or FakeBuilder(), self.root, list(scenarios), repeat, 1, "a note", overrides or {}, False,
            now=lambda: datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc), loadavg=loadavg, log=self.log.append,
        )

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


if __name__ == "__main__":
    unittest.main()
