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


def trace_file(events):
    return {"traceEvents": events, "beginningOfTime": 1790000000000000}


def unit(main="/repo/src/a.cpp", extra=(), total=1_000_000, frontend=800_000, backend=150_000):
    """One translation unit: main file, a header that includes <format>, two nested instantiations."""
    events = [
        ev("ExecuteCompiler", 0, total, ""),
        ev("Frontend", 10, frontend, ""),
        ev("Backend", 810_000, backend, ""),
        ev("Source", 20, 700_000, main),
        ev("Source", 100, 300_000, "/repo/include/x.hpp"),
        ev("Source", 200, 100_000, "/usr/include/c++/14/format"),
        ev("InstantiateClass", 400_000, 50_000, "std::vector<int>"),
        ev("InstantiateFunction", 410_000, 20_000, "std::vector<int>::push_back"),
        ev("Total Source", 0, 400_000, ""),
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "args": {"name": "clang"}},
    ]
    return trace_file(events + list(extra))


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


class AnalyseTraceTests(unittest.TestCase):
    def analyse(self, data):
        return traces.analyse_trace(trace_file_events(data))

    def test_totals_main_file_and_counts(self):
        result = traces.analyse_trace(traces_events(unit()))
        self.assertEqual("/repo/src/a.cpp", result["main"])
        self.assertEqual((1.0, 0.8, 0.15), (result["total_s"], result["frontend_s"], result["backend_s"]))
        self.assertEqual(3, result["source_events"])
        self.assertEqual(2, result["instantiation_events"])

    def test_headers_have_inclusive_and_self_time_and_exclude_the_main_file(self):
        headers = {h: (round(i, 6), round(s, 6)) for h, i, s in traces.analyse_trace(traces_events(unit()))["headers"]}
        self.assertEqual({"/repo/include/x.hpp", "/usr/include/c++/14/format"}, set(headers))
        self.assertEqual((0.3, 0.2), headers["/repo/include/x.hpp"])
        self.assertEqual((0.1, 0.1), headers["/usr/include/c++/14/format"])

    def test_templates_have_self_time_after_nested_instantiations(self):
        templates = {t: (round(i, 6), round(s, 6)) for t, i, s in traces.analyse_trace(traces_events(unit()))["templates"]}
        self.assertEqual((0.05, 0.03), templates["std::vector<int>"])
        self.assertEqual((0.02, 0.02), templates["std::vector<int>::push_back"])

    def test_the_largest_top_level_source_is_the_main_file(self):
        extra = [ev("Source", 1, 10, "/repo/build/cmake_pch.hxx")]
        result = traces.analyse_trace(traces_events(unit(extra=extra)))
        self.assertEqual("/repo/src/a.cpp", result["main"])
        self.assertIn("/repo/build/cmake_pch.hxx", [h for h, *_ in result["headers"]])

    def test_without_executecompiler_the_total_is_frontend_plus_backend(self):
        data = unit()
        data["traceEvents"] = [e for e in data["traceEvents"] if e["name"] != "ExecuteCompiler"]
        self.assertAlmostEqual(0.95, traces.analyse_trace(traces_events(data))["total_s"])

    def test_missing_backend_is_none_and_unusable_traces_have_no_total(self):
        data = unit()
        data["traceEvents"] = [e for e in data["traceEvents"] if e["name"] != "Backend"]
        self.assertIsNone(traces.analyse_trace(traces_events(data))["backend_s"])
        empty = traces.analyse_trace([])
        self.assertIsNone(empty["total_s"])
        self.assertEqual("", empty["main"])

    def test_total_events_and_metadata_are_not_counted_as_includes(self):
        result = traces.analyse_trace(traces_events(unit()))
        self.assertEqual(3, result["source_events"])


def traces_events(data):
    """Events as load_trace returns them (complete events only)."""
    return [e for e in data["traceEvents"] if e.get("ph") == "X"]


def trace_file_events(data):
    return traces_events(data)


class HelperTests(unittest.TestCase):
    def test_clean_path_strips_the_first_matching_root(self):
        self.assertEqual("src/a.cpp", traces.clean_path("/repo/src/a.cpp", ["/repo", "/repo/build"]))
        self.assertEqual("x.cpp.o", traces.clean_path("/repo/build/x.cpp.o", ["/repo/build", "/repo"]))
        self.assertEqual("/usr/include/c++/14/format", traces.clean_path("/usr/include/c++/14/format", ["/repo"]))
        self.assertEqual("/repository/x", traces.clean_path("/repository/x", ["/repo"]))

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

    def test_keeps_only_complete_events(self):
        events = self.load(json.dumps({"traceEvents": [ev("A", 0, 5), {"ph": "M", "name": "x"}, {"ph": "X", "name": "B"}, "junk"]}))
        self.assertEqual(["A"], [e["name"] for e in events])

    def test_bad_files_raise_value_error(self):
        for text in ("", "{not json", "[]", json.dumps({"traceEvents": "no"}), json.dumps({})):
            with self.assertRaises(ValueError, msg=text):
                self.load(text)
        with self.assertRaises(ValueError):
            traces.load_trace("/nonexistent/x.cpp.json")


class SummariseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.build = self.root / "build"
        self.build.mkdir()

    def put(self, name, data):
        path = self.build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return path

    def summary(self):
        return traces.summarise(self.build, self.root)

    def test_totals_and_slowest_files_are_ordered_and_use_repo_relative_paths(self):
        a = unit(main=f"{self.root}/src/a.cpp", total=1_000_000)
        b = unit(main=f"{self.root}/src/b.cpp", total=3_000_000, frontend=2_000_000, backend=900_000)
        self.put("src/a.cpp.json", a)
        self.put("src/b.cpp.json", b)
        summary, _ = self.summary()
        self.assertEqual(2, summary["files"])
        self.assertEqual(4.0, summary["total_s"])
        self.assertEqual(2.8, summary["frontend_s"])
        self.assertAlmostEqual(1.05, summary["backend_s"])
        self.assertEqual(["src/b.cpp", "src/a.cpp"], [f["file"] for f in summary["slowest_files"]])
        self.assertEqual(3.0, summary["slowest_files"][0]["total_s"])

    def test_headers_are_summed_over_translation_units(self):
        self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp"))
        self.put("b.cpp.json", unit(main=f"{self.root}/b.cpp"))
        summary, _ = self.summary()
        header = next(h for h in summary["headers"] if h["header"] == "/usr/include/c++/14/format")
        self.assertEqual({"inclusive_s": 0.2, "self_s": 0.2, "events": 2, "files": 2}, {k: header[k] for k in ("inclusive_s", "self_s", "events", "files")})
        names = [h["header"] for h in summary["headers"]]
        self.assertIn("/repo/include/x.hpp", names)  # outside the temporary root: left as it is
        self.assertEqual(names, sorted(names, key=lambda n: -next(h["self_s"] for h in summary["headers"] if h["header"] == n)))

    def test_header_and_main_paths_inside_the_source_root_are_made_relative(self):
        data = unit(main=f"{self.root}/src/a.cpp")
        for event in data["traceEvents"]:
            if event.get("args", {}).get("detail") == "/repo/include/x.hpp":
                event["args"]["detail"] = f"{self.root}/include/x.hpp"
        self.put("a.cpp.json", data)
        summary, _ = self.summary()
        self.assertIn("include/x.hpp", [h["header"] for h in summary["headers"]])
        self.assertEqual("src/a.cpp", summary["slowest_files"][0]["file"])

    def test_headers_are_ranked_by_self_time_not_by_what_they_include(self):
        umbrella = [
            ev("ExecuteCompiler", 0, 1_000_000),
            ev("Source", 10, 900_000, f"{self.root}/a.cpp"),
            ev("Source", 20, 500_000, f"{self.root}/umbrella.hpp"),  # self 100 ms
            ev("Source", 30, 200_000, f"{self.root}/leaf1.hpp"),
            ev("Source", 300_000, 200_000, f"{self.root}/leaf2.hpp"),
        ]
        self.put("a.cpp.json", trace_file(umbrella))
        summary, _ = self.summary()
        names = [h["header"] for h in summary["headers"]]
        self.assertEqual(["leaf1.hpp", "leaf2.hpp", "umbrella.hpp"], names)
        self.assertEqual(0.5, next(h for h in summary["headers"] if h["header"] == "umbrella.hpp")["inclusive_s"])

    def test_files_counts_translation_units_and_events_counts_inclusions(self):
        twice = [ev("Source", 800_000, 1000, "/repo/include/x.hpp")]
        self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp", extra=twice))
        self.put("b.cpp.json", unit(main=f"{self.root}/b.cpp"))
        header = next(h for h in self.summary()[0]["headers"] if h["header"] == "/repo/include/x.hpp")
        self.assertEqual((3, 2), (header["events"], header["files"]))

    def test_templates_are_ranked_by_self_time_with_counts(self):
        self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp"))
        self.put("b.cpp.json", unit(main=f"{self.root}/b.cpp"))
        summary, _ = self.summary()
        top = summary["templates"][0]
        self.assertEqual(("std::vector<int>", 2, 0.1, 0.06), (top["name"], top["count"], top["inclusive_s"], top["self_s"]))

    def test_counts_for_the_noise_free_figures(self):
        self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp"))
        self.put("b.cpp.json", unit(main=f"{self.root}/b.cpp"))
        summary, _ = self.summary()
        self.assertEqual((6, 4), (summary["source_events"], summary["instantiation_events"]))

    def test_unusable_traces_are_counted_not_fatal(self):
        self.put("good.cpp.json", unit(main=f"{self.root}/good.cpp"))
        self.put("broken.cpp.json", "{truncated")
        self.put("empty.cpp.json", {"traceEvents": []})
        self.put("notatrace.cpp.json", {"hello": 1})
        summary, _ = self.summary()
        self.assertEqual((1, 3), (summary["files"], summary["unreadable"]))

    def test_the_lists_are_capped(self):
        with mock.patch.multiple(traces, TOP_FILES=1, TOP_HEADERS=1, TOP_TEMPLATES=1):
            self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp"))
            self.put("b.cpp.json", unit(main=f"{self.root}/b.cpp", total=2_000_000))
            summary, _ = self.summary()
        self.assertEqual((1, 1, 1), (len(summary["slowest_files"]), len(summary["headers"]), len(summary["templates"])))
        self.assertEqual("b.cpp", summary["slowest_files"][0]["file"])

    def test_a_trace_without_source_events_is_named_after_its_file(self):
        data = trace_file([ev("ExecuteCompiler", 0, 100_000), ev("Frontend", 0, 80_000)])
        self.put("src/odd.cpp.json", data)
        summary, _ = self.summary()
        self.assertEqual("src/odd.cpp", summary["slowest_files"][0]["file"])

    def test_output_is_deterministic(self):
        for i in range(4):
            self.put(f"t{i}.cpp.json", unit(main=f"{self.root}/t{i}.cpp", total=1_000_000 + 10 * i))
        first = json.dumps(self.summary()[0], sort_keys=True)
        second = json.dumps(self.summary()[0], sort_keys=True)
        self.assertEqual(first, second)

    def test_long_template_names_are_clipped(self):
        self.put("a.cpp.json", unit(main=f"{self.root}/a.cpp", extra=[ev("InstantiateClass", 600_000, 1000, "T<" + "x" * 1000 + ">")]))
        summary, _ = self.summary()
        self.assertTrue(all(len(t["name"]) <= traces.MAX_NAME_CHARS for t in summary["templates"]))


class RawTraceTests(unittest.TestCase):
    def test_only_the_heaviest_are_copied_with_flattened_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build, out = root / "build", root / "out"
            (build / "src").mkdir(parents=True)
            for name, total in (("a", 1_000_000), ("b", 3_000_000), ("c", 2_000_000)):
                (build / "src" / f"{name}.cpp.json").write_text(json.dumps(unit(main=f"{root}/{name}.cpp", total=total)))
            with mock.patch.object(traces, "RAW_TRACES", 2):
                summary, heaviest = traces.summarise(build, root)
                traces.copy_raw_traces(heaviest, build, out)
            self.assertEqual(["src__b.cpp.json", "src__c.cpp.json"], sorted(p.name for p in (out / "traces").iterdir()))
            self.assertEqual((build / "src" / "b.cpp.json").read_text(), (out / "traces" / "src__b.cpp.json").read_text())


class MainTests(unittest.TestCase):
    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = traces.main(list(args))
        return code, out.getvalue()

    def test_writes_the_summary_and_traces_and_always_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "build" / "a.cpp.json").write_text(json.dumps(unit(main=f"{root}/a.cpp")))
            code, printed = self.run_main("--build-dir", str(root / "build"), "--source-root", str(root), "--out", str(root / "out"))
            self.assertEqual(0, code)
            data = json.loads((root / "out" / "clang-traces.json").read_text())
            self.assertEqual(("clang-trace-summary", 1, 1), (data["kind"], data["schema_version"], data["files"]))
            self.assertTrue((root / "out" / "traces" / "a.cpp.json").exists())
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
