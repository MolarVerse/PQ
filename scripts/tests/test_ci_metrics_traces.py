import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

METRICS = Path(__file__).resolve().parents[2] / ".github" / "ci-metrics"
SPEC = importlib.util.spec_from_file_location("ci_metrics_traces", METRICS / "summarise_traces.py")
traces = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = traces
SPEC.loader.exec_module(traces)

ROOT = "/repo"


def ev(name, ts, dur, detail=None, tid=0):
    event = {"pid": 1, "tid": tid, "ph": "X", "ts": ts, "dur": dur, "name": name}
    event["args"] = {} if detail is None else {"detail": detail}
    return event


def src(ts, dur, path, tid=0):
    """An include as clang 20 writes it: an async begin and end pair, adjacent, id 0."""
    base = {"pid": 1, "tid": tid, "cat": "Source", "id": 0, "name": "Source"}
    return [
        dict(base, ph="b", ts=ts, args={"detail": path}),
        dict(base, ph="e", ts=ts + dur),
    ]


def trace_file(events):
    return {"traceEvents": events, "beginningOfTime": 1790000000000000}


def unit(extra=(), total=1_000_000, frontend=800_000, backend=150_000):
    """One translation unit: a header that includes <format>, two nested instantiations.

    Like clang, the nested include is written before the include that contains it.
    """
    events = [
        ev("ExecuteCompiler", 0, total, ""),
        ev("Frontend", 10, frontend, ""),
        ev("Backend", 810_000, backend, ""),
        *src(200, 100_000, "/usr/include/c++/14/format"),
        *src(100, 300_000, "/repo/include/x.hpp"),
        ev("InstantiateClass", 400_000, 50_000, "std::vector<int>"),
        ev("InstantiateFunction", 410_000, 20_000, "std::vector<int>::push_back"),
        ev("Total Source", 0, 400_000, ""),
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "clang"}},
    ]
    return trace_file(events + list(extra))


SAMPLE = Path(__file__).resolve().parent / "data" / "clang_trace_sample.json"


class SelfTimeTests(unittest.TestCase):
    def selfs(self, events):
        return {(e["name"], e["ts"]): (depth, s) for e, depth, s in traces.with_self_times(events)}

    def test_children_are_taken_out_of_the_parent(self):
        result = self.selfs([ev("Source", 0, 100), ev("Source", 10, 20), ev("Source", 40, 10), ev("Source", 200, 5)])
        self.assertEqual((0, 70.0), result[("Source", 0)])
        self.assertEqual((1, 20.0), result[("Source", 10)])
        self.assertEqual((1, 10.0), result[("Source", 40)])
        self.assertEqual((0, 5.0), result[("Source", 200)])

    def test_only_direct_children_are_subtracted(self):
        result = self.selfs([ev("Source", 0, 100), ev("Source", 10, 50), ev("Source", 20, 10)])
        self.assertEqual(50.0, result[("Source", 0)][1])
        self.assertEqual(40.0, result[("Source", 10)][1])
        self.assertEqual(10.0, result[("Source", 20)][1])

    def test_a_child_ending_a_microsecond_late_is_clamped(self):
        result = self.selfs([ev("Source", 0, 100), ev("Source", 50, 51)])
        self.assertEqual(50.0, result[("Source", 0)][1])

    def test_threads_do_not_nest_into_each_other(self):
        result = self.selfs([ev("Source", 0, 100, tid=0), ev("Source", 10, 20, tid=1)])
        self.assertEqual((0, 100.0), result[("Source", 0)])
        self.assertEqual((0, 20.0), result[("Source", 10)])

    def test_input_order_does_not_matter_and_nothing_goes_negative(self):
        events = [ev("Source", 10, 20), ev("Source", 0, 100), ev("Source", 12, 100)]
        result = self.selfs(events)
        self.assertTrue(all(value >= 0 for _, value in result.values()))
        self.assertEqual(result, self.selfs(list(reversed(events))))


def events_of(data):
    return traces.complete_events(data["traceEvents"])


class CompleteEventsTests(unittest.TestCase):
    def test_async_pairs_become_complete_events_with_the_begin_detail(self):
        events = traces.complete_events(src(100, 50, "/a.hpp") + [ev("Frontend", 0, 10)])
        self.assertEqual([("Source", 100, 50, "/a.hpp"), ("Frontend", 0, 10, "")], [(e["name"], e["ts"], e["dur"], traces.detail(e)) for e in events])

    def test_nested_pairs_in_clangs_order_and_in_proper_nesting_both_pair_correctly(self):
        clang_order = src(20, 10, "/inner.hpp") + src(10, 40, "/outer.hpp")
        proper = [src(10, 40, "/outer.hpp")[0], *src(20, 10, "/inner.hpp"), src(10, 40, "/outer.hpp")[1]]
        for raw in (clang_order, proper):
            events = {traces.detail(e): (e["ts"], e["dur"]) for e in traces.complete_events(raw)}
            self.assertEqual({"/inner.hpp": (20, 10), "/outer.hpp": (10, 40)}, events)

    def test_unmatched_and_inconsistent_events_are_dropped(self):
        begin, end = src(10, 5, "/a.hpp")
        backwards = src(100, -5, "/b.hpp")
        self.assertEqual([], traces.complete_events([end]))
        self.assertEqual([], traces.complete_events([begin]))
        self.assertEqual([], traces.complete_events(backwards))
        self.assertEqual([], traces.complete_events(["junk", {"ph": "b"}, {"ph": "X", "name": "x", "ts": "1", "dur": 1}, {"ph": "X", "name": "x", "ts": 1}]))

    def test_threads_and_categories_do_not_pair_with_each_other(self):
        a_begin, _ = src(10, 5, "/a.hpp", tid=1)
        _, b_end = src(10, 5, "/b.hpp", tid=2)
        self.assertEqual([], traces.complete_events([a_begin, b_end]))
        other_cat = dict(src(10, 5, "/c.hpp")[1], cat="Other")
        self.assertEqual([], traces.complete_events([src(10, 5, "/c.hpp")[0], other_cat]))

    def test_older_complete_source_events_give_the_same_analysis(self):
        async_form = unit()
        complete_form = unit()
        complete_form["traceEvents"] = [e for e in complete_form["traceEvents"] if e.get("cat") != "Source"] + [
            ev("Source", 100, 300_000, "/repo/include/x.hpp"),
            ev("Source", 200, 100_000, "/usr/include/c++/14/format"),
        ]
        self.assertEqual(
            sorted(traces.analyse_trace(events_of(async_form))["headers"]),
            sorted(traces.analyse_trace(events_of(complete_form))["headers"]),
        )


class AnalyseTraceTests(unittest.TestCase):
    def test_totals_and_counts(self):
        result = traces.analyse_trace(events_of(unit()))
        self.assertEqual((1.0, 0.8, 0.15), (result["total_s"], result["frontend_s"], result["backend_s"]))
        self.assertEqual(2, result["source_events"])
        self.assertEqual(2, result["instantiation_events"])

    def test_headers_have_inclusive_and_self_time(self):
        headers = {h: (round(i, 6), round(s, 6)) for h, i, s in traces.analyse_trace(events_of(unit()))["headers"]}
        self.assertEqual({"/repo/include/x.hpp", "/usr/include/c++/14/format"}, set(headers))
        self.assertEqual((0.3, 0.2), headers["/repo/include/x.hpp"])
        self.assertEqual((0.1, 0.1), headers["/usr/include/c++/14/format"])

    def test_there_is_no_main_file_event_so_every_top_level_include_is_a_header(self):
        extra = src(500_000, 1000, "/repo/include/y.hpp")
        headers = [h for h, *_ in traces.analyse_trace(events_of(unit(extra=extra)))["headers"]]
        self.assertIn("/repo/include/y.hpp", headers)
        self.assertIn("/repo/include/x.hpp", headers)

    def test_templates_have_self_time_after_nested_instantiations(self):
        templates = {t: (round(i, 6), round(s, 6)) for t, i, s in traces.analyse_trace(events_of(unit()))["templates"]}
        self.assertEqual((0.05, 0.03), templates["std::vector<int>"])
        self.assertEqual((0.02, 0.02), templates["std::vector<int>::push_back"])

    def test_without_executecompiler_the_total_is_frontend_plus_backend(self):
        data = unit()
        data["traceEvents"] = [e for e in data["traceEvents"] if e["name"] != "ExecuteCompiler"]
        self.assertAlmostEqual(0.95, traces.analyse_trace(events_of(data))["total_s"])

    def test_missing_backend_is_none_and_unusable_traces_have_no_total(self):
        data = unit()
        data["traceEvents"] = [e for e in data["traceEvents"] if e["name"] != "Backend"]
        self.assertIsNone(traces.analyse_trace(events_of(data))["backend_s"])
        self.assertIsNone(traces.analyse_trace([])["total_s"])

    def test_total_events_and_metadata_are_not_counted_as_includes(self):
        self.assertEqual(2, traces.analyse_trace(events_of(unit()))["source_events"])


class RealTraceTests(unittest.TestCase):
    """A trimmed trace written by clang 20.1.2 for src/constraints/mShake.cpp."""

    def test_the_sample_is_read_as_clang_wrote_it(self):
        raw = json.loads(SAMPLE.read_text())["traceEvents"]
        by_name = {e["name"]: e for e in raw if e.get("ph") == "X"}
        result = traces.analyse_trace(traces.load_trace(SAMPLE))
        self.assertEqual(75, result["source_events"])
        self.assertEqual(40, result["instantiation_events"])
        # the totals of the file itself agree with what is derived from the events
        self.assertAlmostEqual(by_name["Total ExecuteCompiler"]["dur"] / 1e6, result["total_s"])
        self.assertAlmostEqual(by_name["Total Frontend"]["dur"] / 1e6, result["frontend_s"], places=5)
        self.assertAlmostEqual(by_name["Total Backend"]["dur"] / 1e6, result["backend_s"], places=5)

    def test_headers_of_the_sample_nest_and_have_sane_self_times(self):
        result = traces.analyse_trace(traces.load_trace(SAMPLE))
        headers = {}
        for header, inclusive, own in result["headers"]:
            self.assertGreaterEqual(own, 0)
            self.assertLessEqual(own, inclusive + 1e-9)
            headers[header] = (inclusive, own)
        outer = "/home/runner/work/PQ/PQ/include/constraints/mShake.hpp"
        inner = "/home/runner/work/PQ/PQ/include/molsys/atom.hpp"
        self.assertIn(outer, headers)
        self.assertAlmostEqual(0.439364, headers[outer][0], places=6)
        self.assertLess(headers[outer][1], headers[outer][0])  # it includes other headers
        self.assertIn(inner, headers)

    def test_a_directory_with_the_sample_is_summarised_with_clean_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "build" / "src" / "constraints" / "CMakeFiles" / "constraints.dir" / "mShake.cpp.json"
            target.parent.mkdir(parents=True)
            target.write_text(SAMPLE.read_text())
            summary, _ = traces.summarise(root / "build", "/home/runner/work/PQ/PQ")
        self.assertEqual("src/constraints/mShake.cpp", summary["slowest_files"][0]["file"])
        names = [h["header"] for h in summary["headers"]]
        self.assertIn("include/constraints/mShake.hpp", names)
        self.assertTrue(all(".." not in name.split("/") for name in names))


class HelperTests(unittest.TestCase):
    def test_clean_path_strips_the_first_matching_root(self):
        self.assertEqual("src/a.cpp", traces.clean_path("/repo/src/a.cpp", ["/repo", "/repo/build"]))
        self.assertEqual("x.cpp.o", traces.clean_path("/repo/build/x.cpp.o", ["/repo/build", "/repo"]))
        self.assertEqual("/usr/include/c++/14/format", traces.clean_path("/usr/include/c++/14/format", ["/repo"]))
        self.assertEqual("/repository/x", traces.clean_path("/repository/x", ["/repo"]))

    def test_clean_path_normalises_dot_dot_segments(self):
        self.assertEqual("/usr/include/c++/14/filesystem", traces.clean_path("/usr/lib/gcc/x86_64-linux-gnu/14/../../../../include/c++/14/filesystem", ["/repo"]))
        self.assertEqual("include/a/b.hpp", traces.clean_path("/repo/include/a/x/../b.hpp", ["/repo"]))

    def test_tu_name_maps_the_object_directory_back_to_the_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            for trace, expected in (
                ("src/constraints/CMakeFiles/constraints.dir/mShake.cpp.json", "src/constraints/mShake.cpp"),
                ("apps/CMakeFiles/PQ.dir/validation.cpp.json", "apps/validation.cpp"),
                ("CMakeFiles/top.dir/main.cpp.json", "main.cpp"),
                ("external/googletest/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.json", "external/googletest/googletest/src/gtest-all.cc"),
                ("odd/place.cpp.json", "odd/place.cpp"),
            ):
                self.assertEqual(expected, traces.tu_name(build / trace, build), trace)
            self.assertEqual("elsewhere.cpp", traces.tu_name("/somewhere/else/elsewhere.cpp.json", build))

    def test_clip_limits_length_and_hides_control_characters(self):
        self.assertEqual("a b", traces.clip("a\nb"))
        clipped = traces.clip("x" * 500)
        self.assertEqual(traces.MAX_NAME_CHARS, len(clipped))
        self.assertTrue(clipped.endswith("…"))

    def test_only_trace_file_names_are_found(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            for name in ("a.cpp.json", "sub/b.cc.json", "sub/c.cxx.json", "d.c.json", "compile_commands.json", "e.json", "f.cpp.json.bak"):
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            found = [str(p.relative_to(build)) for p in traces.find_traces(build)]
            self.assertEqual(["a.cpp.json", "d.c.json", "sub/b.cc.json", "sub/c.cxx.json"], found)


class LoadTraceTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.cpp.json"
            path.write_text(text)
            return traces.load_trace(path)

    def test_keeps_complete_events_and_pairs_async_ones(self):
        raw = [ev("A", 0, 5), {"ph": "M", "name": "x"}, {"ph": "X", "name": "B"}, "junk", *src(10, 5, "/h.hpp")]
        events = self.load(json.dumps({"traceEvents": raw}))
        self.assertEqual(["A", "Source"], [e["name"] for e in events])

    def test_bad_files_raise_value_error(self):
        for text in ("", "{not json", "[]", json.dumps({"traceEvents": "no"}), json.dumps({})):
            with self.assertRaises(ValueError, msg=text):
                self.load(text)
        with self.assertRaises(ValueError):
            traces.load_trace("/nonexistent/x.cpp.json")


def object_path(name, directory="src"):
    """Where CMake and clang put the trace of `directory/name`."""
    return f"{directory}/CMakeFiles/lib.dir/{name}.json"


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.build = self.root / "build"
        self.build.mkdir()

    def put(self, name, data, directory="src"):
        path = self.build / object_path(name, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return path

    def summary(self):
        return traces.summarise(self.build, self.root)

    def test_totals_and_slowest_files_are_ordered_and_named_after_the_source(self):
        self.put("a.cpp", unit(total=1_000_000))
        self.put("b.cpp", unit(total=3_000_000, frontend=2_000_000, backend=900_000))
        summary, _ = self.summary()
        self.assertEqual(2, summary["files"])
        self.assertEqual(4.0, summary["total_s"])
        self.assertEqual(2.8, summary["frontend_s"])
        self.assertAlmostEqual(1.05, summary["backend_s"])
        self.assertEqual(["src/b.cpp", "src/a.cpp"], [f["file"] for f in summary["slowest_files"]])
        self.assertEqual(3.0, summary["slowest_files"][0]["total_s"])

    def test_headers_are_summed_over_translation_units(self):
        self.put("a.cpp", unit())
        self.put("b.cpp", unit())
        summary, _ = self.summary()
        header = next(h for h in summary["headers"] if h["header"] == "/usr/include/c++/14/format")
        self.assertEqual({"inclusive_s": 0.2, "self_s": 0.2, "events": 2, "files": 2}, {k: header[k] for k in ("inclusive_s", "self_s", "events", "files")})
        selfs = [h["self_s"] for h in summary["headers"]]
        self.assertEqual(selfs, sorted(selfs, reverse=True))

    def test_headers_inside_the_source_root_are_made_relative(self):
        data = unit(extra=src(600_000, 1000, f"{self.root}/include/y.hpp"))
        self.put("a.cpp", data)
        summary, _ = self.summary()
        self.assertIn("include/y.hpp", [h["header"] for h in summary["headers"]])
        self.assertIn("/repo/include/x.hpp", [h["header"] for h in summary["headers"]])  # outside the root: unchanged

    def test_headers_are_ranked_by_self_time_not_by_what_they_include(self):
        umbrella = [
            ev("ExecuteCompiler", 0, 1_000_000),
            *src(30, 200_000, f"{self.root}/leaf1.hpp"),
            *src(300_000, 200_000, f"{self.root}/leaf2.hpp"),
            *src(20, 500_000, f"{self.root}/umbrella.hpp"),  # self 100 ms
        ]
        self.put("a.cpp", trace_file(umbrella))
        summary, _ = self.summary()
        names = [h["header"] for h in summary["headers"]]
        self.assertEqual(["leaf1.hpp", "leaf2.hpp", "umbrella.hpp"], names)
        self.assertEqual(0.5, next(h for h in summary["headers"] if h["header"] == "umbrella.hpp")["inclusive_s"])

    def test_files_counts_translation_units_and_events_counts_inclusions(self):
        twice = src(800_000, 1000, "/repo/include/x.hpp")
        self.put("a.cpp", unit(extra=twice))
        self.put("b.cpp", unit())
        header = next(h for h in self.summary()[0]["headers"] if h["header"] == "/repo/include/x.hpp")
        self.assertEqual((3, 2), (header["events"], header["files"]))

    def test_templates_are_ranked_by_self_time_with_counts(self):
        self.put("a.cpp", unit())
        self.put("b.cpp", unit())
        summary, _ = self.summary()
        top = summary["templates"][0]
        self.assertEqual(("std::vector<int>", 2, 0.1, 0.06), (top["name"], top["count"], top["inclusive_s"], top["self_s"]))

    def test_counts_for_the_noise_free_figures(self):
        self.put("a.cpp", unit())
        self.put("b.cpp", unit())
        summary, _ = self.summary()
        self.assertEqual((4, 4), (summary["source_events"], summary["instantiation_events"]))

    def test_unusable_traces_are_counted_not_fatal(self):
        self.put("good.cpp", unit())
        self.put("broken.cpp", "{truncated")
        self.put("empty.cpp", {"traceEvents": []})
        self.put("notatrace.cpp", {"hello": 1})
        summary, _ = self.summary()
        self.assertEqual((1, 3), (summary["files"], summary["unreadable"]))

    def test_the_lists_are_capped(self):
        with mock.patch.multiple(traces, TOP_FILES=1, TOP_HEADERS=1, TOP_TEMPLATES=1):
            self.put("a.cpp", unit())
            self.put("b.cpp", unit(total=2_000_000))
            summary, _ = self.summary()
        self.assertEqual((1, 1, 1), (len(summary["slowest_files"]), len(summary["headers"]), len(summary["templates"])))
        self.assertEqual("src/b.cpp", summary["slowest_files"][0]["file"])

    def test_output_is_deterministic(self):
        for i in range(4):
            self.put(f"t{i}.cpp", unit(total=1_000_000 + 10 * i))
        first = json.dumps(self.summary()[0], sort_keys=True)
        second = json.dumps(self.summary()[0], sort_keys=True)
        self.assertEqual(first, second)

    def test_long_template_names_are_clipped(self):
        self.put("a.cpp", unit(extra=[ev("InstantiateClass", 600_000, 1000, "T<" + "x" * 1000 + ">")]))
        summary, _ = self.summary()
        self.assertTrue(all(len(t["name"]) <= traces.MAX_NAME_CHARS for t in summary["templates"]))


class RawTraceTests(unittest.TestCase):
    def test_only_the_heaviest_are_copied_with_flattened_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build, out = root / "build", root / "out"
            for name, total in (("a", 1_000_000), ("b", 3_000_000), ("c", 2_000_000)):
                path = build / object_path(f"{name}.cpp")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(unit(total=total)))
            with mock.patch.object(traces, "RAW_TRACES", 2):
                summary, heaviest = traces.summarise(build, root)
                traces.copy_raw_traces(heaviest, build, out)
            names = sorted(p.name for p in (out / "traces").iterdir())
            self.assertEqual(["src__CMakeFiles__lib.dir__b.cpp.json", "src__CMakeFiles__lib.dir__c.cpp.json"], names)
            source = build / object_path("b.cpp")
            self.assertEqual(source.read_text(), (out / "traces" / names[0]).read_text())


class MainTests(unittest.TestCase):
    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = traces.main(list(args))
        return code, out.getvalue()

    def test_writes_the_summary_and_traces_and_always_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "build" / object_path("a.cpp")
            trace.parent.mkdir(parents=True)
            trace.write_text(json.dumps(unit()))
            code, printed = self.run_main("--build-dir", str(root / "build"), "--source-root", str(root), "--out", str(root / "out"))
            self.assertEqual(0, code)
            data = json.loads((root / "out" / "clang-traces.json").read_text())
            self.assertEqual(("clang-trace-summary", 1, 1), (data["kind"], data["schema_version"], data["files"]))
            self.assertEqual(1, len(list((root / "out" / "traces").iterdir())))
            self.assertIn("1 traces (0 unreadable)", printed)

    def test_no_raw_flag_and_empty_build_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            code, _ = self.run_main("--build-dir", str(root / "build"), "--out", str(root / "out"), "--no-raw")
            self.assertEqual(0, code)
            data = json.loads((root / "out" / "clang-traces.json").read_text())
            self.assertEqual((0, 0.0, []), (data["files"], data["total_s"], data["slowest_files"]))
            self.assertFalse((root / "out" / "traces").exists())

    def test_missing_build_dir_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _ = self.run_main("--build-dir", f"{directory}/nope", "--out", f"{directory}/out")
            self.assertEqual(0, code)
            self.assertEqual(0, json.loads((Path(directory) / "out" / "clang-traces.json").read_text())["files"])


if __name__ == "__main__":
    unittest.main()
