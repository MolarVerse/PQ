import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

METRICS = Path(__file__).resolve().parents[2] / ".github" / "ci-metrics"
SPEC = importlib.util.spec_from_file_location("ci_metrics_clang_report", METRICS / "clang_report.py")
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def header(name, self_s, files=10, inclusive=None):
    return {"header": name, "self_s": self_s, "inclusive_s": inclusive if inclusive is not None else self_s, "events": files, "files": files}


def summary(
    total=1000.0,
    frontend=700.0,
    backend=300.0,
    source_events=1000,
    instantiation_events=2000,
    headers=(),
    files=(),
    templates=(),
    limits=None,
    kind="clang-trace-detail",
):
    return {
        "schema_version": 1,
        "kind": kind,
        "limits": limits or {"files": 100, "headers": 300, "templates": 150},
        "files": 400,
        "unreadable": 0,
        "total_s": total,
        "frontend_s": frontend,
        "backend_s": backend,
        "source_events": source_events,
        "instantiation_events": instantiation_events,
        "slowest_files": list(files),
        "headers": list(headers),
        "templates": list(templates),
    }


def file_entry(name, total, frontend=None, backend=None):
    return {"file": name, "total_s": total, "frontend_s": frontend if frontend is not None else total / 2, "backend_s": backend if backend is not None else total / 2}


def template(name, self_s, count=10):
    return {"name": name, "count": count, "inclusive_s": self_s, "self_s": self_s}


class FormatTests(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual("-", report.format_seconds(None))
        self.assertEqual("4.2 s", report.format_seconds(4.2))
        self.assertEqual("20m 08s", report.format_seconds(1208.4))
        self.assertEqual("-1m 05s", report.format_seconds(-65))
        self.assertEqual("+2.0 s", report.format_signed_seconds(2.04))
        self.assertEqual("-0.5 s", report.format_signed_seconds(-0.5))

    def test_percent_change(self):
        self.assertEqual("+10.0%", report.format_percent_change(100, 110))
        self.assertEqual("-50.0%", report.format_percent_change(100, 50))
        self.assertEqual("n/a", report.format_percent_change(0, 5))

    def test_code_makes_untrusted_names_safe_for_a_table_cell(self):
        self.assertEqual("`a/b`", report.code("a|b"))
        self.assertEqual("`a'b`", report.code("a`b"))
        self.assertEqual("`a b`", report.code("a\nb"))
        clipped = report.code("x" * 500)
        self.assertEqual(report.NAME_CHARS + 2, len(clipped))
        self.assertTrue(clipped.endswith("…`"))
        self.assertEqual("`</details><script>`", report.code("</details><script>"))


class LoadTests(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.json"
            path.write_text(content if isinstance(content, str) else json.dumps(content))
            return report.load_summary(path)

    def test_valid_detail_and_summary_files_load(self):
        self.assertIsNotNone(self.load(summary()))
        self.assertIsNotNone(self.load(summary(kind="clang-trace-summary")))

    def test_anything_else_is_none(self):
        self.assertIsNone(report.load_summary(None))
        self.assertIsNone(report.load_summary("/nonexistent/s.json"))
        self.assertIsNone(self.load("{broken"))
        self.assertIsNone(self.load("[]"))
        self.assertIsNone(self.load(summary(kind="job")))
        for key in ("total_s", "frontend_s", "backend_s", "source_events", "instantiation_events", "files"):
            broken = summary()
            broken[key] = "12"
            self.assertIsNone(self.load(broken), key)
            del broken[key]
            self.assertIsNone(self.load(broken), key)

    def test_bad_list_entries_are_skipped_not_fatal(self):
        data = summary(headers=[header("ok.hpp", 3.0), {"header": "no-number"}, "junk", {"header": 5, "self_s": 1}])
        self.assertEqual(["ok.hpp"], [e["header"] for e in report.entries(data, "headers", "header", ("self_s",))])


class ChangesTests(unittest.TestCase):
    def changes(self, base, pr, scale=1.0, cut=0.0, min_s=0.5, min_rel=0.15):
        return report.changes(base, pr, scale, cut, min_s, min_rel)

    def test_both_thresholds_must_be_met(self):
        rows = self.changes({"a": 10.0, "b": 10.0, "c": 100.0}, {"a": 12.0, "b": 10.3, "c": 108.0})
        self.assertEqual(["a"], [r[0] for r in rows])
        self.assertAlmostEqual(2.0, rows[0][3])
        self.assertAlmostEqual(0.2, rows[0][4])

    def test_a_uniformly_slower_run_is_not_a_change_but_a_real_one_still_shows(self):
        base = {"a": 10.0, "d": 10.0}
        pr = {"a": 15.0, "d": 20.0}  # the run was 1.5 times slower overall
        rows = self.changes(base, pr, scale=1.5)
        self.assertEqual(["d"], [r[0] for r in rows])
        self.assertAlmostEqual(5.0, rows[0][3])
        self.assertAlmostEqual(1 / 3, rows[0][4])

    def test_lighter_items_have_a_negative_delta(self):
        rows = self.changes({"e": 20.0}, {"e": 10.0})
        self.assertAlmostEqual(-10.0, rows[0][3])

    def test_items_new_to_a_full_baseline_list_are_compared_with_its_cut_off(self):
        rows = self.changes({"x": 6.0, "y": 5.0, "z": 7.0}, {"n": 9.0, "m": 5.2}, cut=5.0)
        self.assertEqual(["n"], [r[0] for r in rows])
        self.assertIsNone(rows[0][1])
        self.assertAlmostEqual(4.0, rows[0][3])

    def test_items_absent_from_a_list_that_was_not_full_are_new_from_zero(self):
        self.assertEqual(["n"], [r[0] for r in self.changes({}, {"n": 0.7, "m": 0.3})])

    def test_the_cut_off_scales_with_the_speed_factor(self):
        self.assertEqual([], self.changes({}, {"n": 6.0}, scale=1.5, cut=5.0))  # 6.0 vs 7.5
        self.assertEqual(["n"], [r[0] for r in self.changes({}, {"n": 9.0}, scale=1.5, cut=5.0)])

    def test_order_is_by_delta_then_name_and_deterministic(self):
        rows = self.changes({"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 3.0, "b": 5.0, "c": 3.0})
        self.assertEqual(["b", "a", "c"], [r[0] for r in rows])

    def test_cut_off_needs_a_full_list_with_known_limit(self):
        data = summary(limits={"files": 1, "headers": 2, "templates": 1})
        self.assertEqual(4.0, report.cut_off(data, "headers", {"a": 4.0, "b": 9.0}))
        self.assertEqual(0.0, report.cut_off(data, "headers", {"a": 4.0}))
        self.assertEqual(0.0, report.cut_off({"limits": "?"}, "headers", {"a": 4.0}))
        self.assertEqual(0.0, report.cut_off(data, "headers", {}))


class RenderTests(unittest.TestCase):
    def render(self, pr, base, **kwargs):
        return report.render(pr, base, **kwargs)

    def test_headline_table_has_both_builds_and_the_change(self):
        text = self.render(summary(total=1100.0, source_events=1010), summary(total=1000.0), baseline_sha="abcdef0123")
        self.assertIn(report.MARKER, text)
        self.assertIn("Compared with `dev` `abcdef0`.", text)
        self.assertIn("| Compiler time (CPU, all files) | 16m 40s | 18m 20s | +10.0% |", text)
        # counts depend on runner speed (events under 0.5 ms are left out), so they get no percentage
        self.assertIn("| Header inclusions (events of at least 0.5 ms) | 1,000 | 1,010 | - |", text)
        self.assertIn("| Template instantiation events (at least 0.5 ms) | 2,000 | 2,000 | - |", text)

    def test_headers_that_got_heavier_are_listed_with_files_and_speed_adjustment(self):
        base = summary(headers=[header("include/a.hpp", 10.0, files=30), header("include/same.hpp", 5.0)])
        pr = summary(total=1000.0, headers=[header("include/a.hpp", 14.0, files=38), header("include/same.hpp", 5.1)])
        text = self.render(pr, base)
        self.assertIn("#### Headers that got heavier", text)
        self.assertIn("| `include/a.hpp` | 10.0 s | 14.0 s | +4.0 s (+40%) | ~30 → ~38 |", text)
        self.assertNotIn("include/same.hpp` | 5.0 s", text.split("<details>")[0])

    def test_a_slower_runner_alone_changes_nothing(self):
        base = summary(headers=[header("a.hpp", 10.0), header("b.hpp", 20.0)], files=[file_entry("src/x.cpp", 30.0)])
        pr = summary(total=1500.0, headers=[header("a.hpp", 15.0), header("b.hpp", 30.0)], files=[file_entry("src/x.cpp", 45.0)])
        text = self.render(pr, base)
        self.assertIn("The compiler time of this run was 1.50 times the baseline's", text)
        self.assertIn("None: no header changed by at least 0.5 s and 15%", text)
        self.assertNotIn("Files that changed most", text)

    def test_lighter_headers_files_and_templates_go_into_the_collapsed_part(self):
        base = summary(headers=[header("a.hpp", 20.0)], files=[file_entry("src/x.cpp", 30.0)], templates=[template("T<int>", 4.0)])
        pr = summary(headers=[header("a.hpp", 10.0)], files=[file_entry("src/x.cpp", 40.0)], templates=[template("T<int>", 8.0)])
        text = self.render(pr, base)
        more = text.split("<details><summary>More changes</summary>")[1].split("</details>")[0]
        self.assertIn("**Headers that got lighter**", more)
        self.assertIn("-10.0 s (-50%)", more)
        self.assertIn("**Files that changed most**", more)
        self.assertIn("| `src/x.cpp` | 30.0 s | 40.0 s | +10.0 s (+33%) |", more)
        self.assertIn("**Template instantiations that changed most**", more)
        self.assertIn("| `T<int>` | 4.0 s | 8.0 s | +4.0 s (+100%) |", more)

    def test_new_headers_are_marked(self):
        base = summary(headers=[header("a.hpp", 5.0), header("b.hpp", 6.0)], limits={"files": 100, "headers": 2, "templates": 150})
        pr = summary(headers=[header("big.hpp", 9.0, files=12)])
        text = self.render(pr, base)
        self.assertIn("| `big.hpp` | - | 9.0 s | +4.0 s (new in the list) | ~- → ~12 |", text)

    def test_the_most_changed_are_capped(self):
        base = summary(headers=[header(f"h{i}.hpp", 1.0) for i in range(30)])
        pr = summary(headers=[header(f"h{i}.hpp", 5.0 + i) for i in range(30)])
        text = self.render(pr, base).split("<details>")[0]
        self.assertEqual(report.SHOW_HEAVIER, text.count("| `h"))
        self.assertIn("`h29.hpp`", text)

    def test_the_details_show_this_pull_requests_numbers(self):
        pr = summary(files=[file_entry("src/x.cpp", 12.0, 5.0, 7.0)], headers=[header("a.hpp", 3.0, files=4, inclusive=6.0)], templates=[template("std::vector<int>", 1.5, count=42)])
        text = self.render(pr, summary())
        details = text.split("<details><summary>Slowest files")[1]
        self.assertIn("| `src/x.cpp` | 12.0 s | 5.0 s | 7.0 s |", details)
        self.assertIn("| `a.hpp` | 3.0 s | 6.0 s | 4 |", details)
        self.assertIn("| `std::vector<int>` | 1.5 s | 42 |", details)

    def test_without_a_baseline_only_this_pull_request_is_shown(self):
        text = self.render(summary(headers=[header("a.hpp", 3.0)]), None)
        self.assertIn("No baseline yet", text)
        self.assertIn("| Compiler time (CPU, all files) | 16m 40s |", text)
        self.assertNotIn("Headers that got heavier", text)
        self.assertNotIn("Compared with", text)
        self.assertIn("`a.hpp`", text)

    def test_html_blocks_and_tables_are_separated_by_a_blank_line(self):
        for base in (None, summary()):
            lines = self.render(summary(headers=[header("a.hpp", 3.0)]), base).splitlines()
            for number, line in enumerate(lines):
                if line.startswith("<details>"):
                    self.assertEqual("", lines[number - 1], line)

    def test_the_run_link_and_the_notes_are_included(self):
        text = self.render(summary(), summary(), run_url="https://example.invalid/run/1")
        self.assertIn("[Workflow run](https://example.invalid/run/1)", text)
        self.assertIn("never a failing check", text)
        self.assertIn("depend on runner speed too", text)

    def test_hostile_names_cannot_break_the_tables_or_the_details_block(self):
        evil = "x|y`z\n</details>| | |"
        pr = summary(headers=[header(evil, 9.0)], files=[file_entry(evil, 9.0)], templates=[template(evil, 9.0)])
        text = self.render(pr, summary())
        blocks, current = [], []
        for line in text.splitlines():
            if line.startswith("|"):
                current.append(line)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
        self.assertGreaterEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual({block[0].count("|")}, {line.count("|") for line in block}, block)
        outside_code = re.sub(r"`[^`\n]*`", "", text)
        self.assertEqual(outside_code.count("<details>"), outside_code.count("</details>"))
        self.assertIn("x/y'z", text)

    def test_output_is_deterministic(self):
        pr = summary(headers=[header(f"h{i}.hpp", 5.0 + i % 3) for i in range(20)])
        base = summary(headers=[header(f"h{i}.hpp", 1.0) for i in range(20)])
        self.assertEqual(self.render(pr, base), self.render(pr, base))


def inc(pairs=1000, project=300, digest="d1", fan_in=None):
    return {"schema_version": 1, "kind": "ninja-includes-detail", "include_pairs": pairs, "project_pairs": project, "digest": digest, "fan_in": fan_in or {}}


class IncludeGraphTests(unittest.TestCase):
    def render(self, includes, base_includes, **kwargs):
        return report.render(summary(), summary(), includes=includes, base_includes=base_includes, **kwargs)

    def section(self, text):
        return text.split("#### Include graph")[1].split("#### Headers that")[0]

    def test_load_includes_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "i.json"
            path.write_text(json.dumps(inc(fan_in={"a.hpp": 3, "bad": "x", "neg": -1})))
            loaded = report.load_includes(path)
            self.assertEqual({"a.hpp": 3}, loaded["fan_in"])
            for content in ("{broken", "[]", json.dumps(dict(inc(), kind="job")), json.dumps(dict(inc(), include_pairs="1")), json.dumps(dict(inc(), fan_in=[]))):
                path.write_text(content)
                self.assertIsNone(report.load_includes(path), content)
        self.assertIsNone(report.load_includes(None))
        self.assertIsNone(report.load_includes("/nonexistent/i.json"))

    def test_an_identical_graph_says_so_and_needs_no_table(self):
        text = self.section(self.render(inc(digest="same", fan_in={"a.hpp": 1}), inc(digest="same", fan_in={"a.hpp": 1})))
        self.assertIn("The include graph is identical to dev's (1,000 include pairs, 300 of them project headers).", text)
        self.assertNotIn("| Header |", text)

    def test_changed_fan_in_is_listed_exactly_largest_first_with_new_and_removed_headers(self):
        base = inc(1000, 300, "d1", {"include/a.hpp": 120, "include/b.hpp": 10, "include/same.hpp": 7})
        now = inc(1030, 330, "d2", {"include/a.hpp": 150, "include/c.hpp": 5, "include/same.hpp": 7})
        text = self.section(self.render(now, base))
        self.assertIn("| Include pairs (translation unit, header) | 1,000 | 1,030 | +30 (+3.0%) |", text)
        self.assertIn("| ... of which project headers | 300 | 330 | +30 |", text)
        self.assertIn("(3 in total, the 3 largest changes)", text)
        rows = [line for line in text.splitlines() if line.startswith("| `")]
        self.assertEqual(
            ["| `include/a.hpp` | 120 | 150 | +30 |", "| `include/b.hpp` | 10 | - | -10 |", "| `include/c.hpp` | - | 5 | +5 |"],
            rows,
        )
        self.assertNotIn("same.hpp", text)

    def test_any_change_counts_there_is_no_threshold(self):
        text = self.section(self.render(inc(digest="d2", fan_in={"a.hpp": 101}), inc(digest="d1", fan_in={"a.hpp": 100})))
        self.assertIn("| `a.hpp` | 100 | 101 | +1 |", text)

    def test_the_list_is_capped_and_says_how_many_changed(self):
        base = inc(digest="d1", fan_in={f"h{i}.hpp": 1 for i in range(30)})
        now = inc(digest="d2", fan_in={f"h{i}.hpp": 1 + i for i in range(30)})
        text = self.section(self.render(now, base))
        self.assertIn("(29 in total, the 10 largest changes)", text)
        rows = [line for line in text.splitlines() if line.startswith("| `")]
        self.assertEqual(report.SHOW_FAN_IN, len(rows))
        self.assertTrue(rows[0].startswith("| `h29.hpp`"))

    def test_a_different_graph_without_a_project_fan_in_change_is_explained(self):
        text = self.section(self.render(inc(1010, 300, "d2", {"a.hpp": 1}), inc(1000, 300, "d1", {"a.hpp": 1})))
        self.assertIn("No project header changed the number of files that include it", text)

    def test_without_a_baseline_only_the_totals_are_shown(self):
        text = report.render(summary(), None, includes=inc(5000, 700))
        self.assertIn("| Include pairs (translation unit, header) | 5,000 |", text)
        self.assertIn("| ... of which project headers | 700 |", text)

    def test_without_include_data_there_is_no_section(self):
        self.assertNotIn("Include graph", report.render(summary(), summary()))
        self.assertNotIn("Include graph", report.render(summary(), None))

    def test_exact_fan_in_replaces_the_approximate_files_column(self):
        base = summary(headers=[header("include/a.hpp", 10.0, files=30), header("/usr/include/c++/14/format", 5.0, files=100)])
        pr = summary(headers=[header("include/a.hpp", 14.0, files=38), header("/usr/include/c++/14/format", 9.0, files=90)])
        text = report.render(pr, base, includes=inc(fan_in={"include/a.hpp": 41}), base_includes=inc(fan_in={"include/a.hpp": 40}))
        heavier = text.split("#### Headers that got heavier")[1].split("<details>")[0]
        self.assertIn("| `include/a.hpp` | 10.0 s | 14.0 s | +4.0 s (+40%) | 40 → 41 |", heavier)
        self.assertIn("| `/usr/include/c++/14/format` | 5.0 s | 9.0 s | +4.0 s (+80%) | ~100 → ~90 |", heavier)

    def test_hostile_header_names_are_made_safe(self):
        evil = "x|y`z\n</details>"
        text = self.render(inc(digest="d2", fan_in={evil: 5}), inc(digest="d1", fan_in={}))
        row = [line for line in self.section(text).splitlines() if line.startswith("| `x")][0]
        self.assertEqual(5, row.count("|"))
        self.assertIn("x/y'z", row)

    def test_main_passes_the_include_files_through(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (("pr.json", summary()), ("base.json", summary()), ("i.json", inc(digest="d2", fan_in={"a.hpp": 2})), ("bi.json", inc(digest="d1", fan_in={"a.hpp": 1}))):
                (root / name).write_text(json.dumps(data))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                report.main(["--pr", str(root / "pr.json"), "--baseline", str(root / "base.json"), "--includes", str(root / "i.json"), "--baseline-includes", str(root / "bi.json"), "--out", str(root / "r.md")])
            self.assertIn("| `a.hpp` | 1 | 2 | +1 |", (root / "r.md").read_text())


class FitAndUnavailableTests(unittest.TestCase):
    def test_a_huge_comment_loses_its_details_first(self):
        pr = summary(files=[file_entry(f"src/{'x' * 100}{i}.cpp", 9.0) for i in range(20)], headers=[header("h" * 120 + str(i), 5.0) for i in range(30)])
        with unittest.mock.patch.object(report, "MAX_COMMENT_CHARS", 3000):
            text = report.render(pr, summary())
        self.assertLessEqual(len(text), 3000)
        self.assertNotIn("<details><summary>Slowest files", text)
        self.assertIn(report.MARKER, text)

    def test_an_absurd_comment_is_truncated(self):
        self.assertLessEqual(len(report.fit("y" * 70_000)), report.MAX_COMMENT_CHARS)
        self.assertEqual("short\n", report.fit("short\n"))

    def test_unavailable_messages(self):
        self.assertIn("build failed", report.render_unavailable("failure"))
        self.assertIn("cancelled", report.render_unavailable("cancelled"))
        other = report.render_unavailable("success", "https://example.invalid/run/2")
        self.assertIn("did not produce usable", other)
        self.assertIn("[Workflow run](https://example.invalid/run/2)", other)
        for status in ("failure", "cancelled", "success", ""):
            self.assertIn(report.MARKER, report.render_unavailable(status))


class MainTests(unittest.TestCase):
    def run_main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = report.main(list(args))
        return code, out.getvalue()

    def files(self, directory, pr=None, base=None):
        root = Path(directory)
        paths = {}
        for name, data in (("pr.json", pr), ("base.json", base)):
            if data is not None:
                (root / name).write_text(data if isinstance(data, str) else json.dumps(data))
                paths[name] = str(root / name)
        return root, paths

    def test_with_and_without_a_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root, paths = self.files(directory, summary(), summary())
            code, printed = self.run_main("--pr", paths["pr.json"], "--baseline", paths["base.json"], "--baseline-sha", "1234567890", "--out", str(root / "r.md"))
            self.assertEqual(0, code)
            self.assertIn("baseline found", printed)
            self.assertIn("Compared with `dev` `1234567`.", (root / "r.md").read_text())
            self.run_main("--pr", paths["pr.json"], "--baseline", str(root / "missing.json"), "--out", str(root / "r2.md"))
            self.assertIn("No baseline yet", (root / "r2.md").read_text())

    def test_unusable_pr_data_still_writes_a_comment_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root, paths = self.files(directory, "{broken")
            code, printed = self.run_main("--pr", paths["pr.json"], "--status", "failure", "--out", str(root / "r.md"))
            self.assertEqual(0, code)
            self.assertIn("no trace data", printed)
            self.assertIn("build failed", (root / "r.md").read_text())
            code, _ = self.run_main("--pr", str(root / "nothing.json"), "--out", str(root / "r3.md"))
            self.assertEqual(0, code)
            self.assertIn("did not produce usable", (root / "r3.md").read_text())


if __name__ == "__main__":
    unittest.main()
