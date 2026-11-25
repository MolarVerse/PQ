#!/usr/bin/env python3
"""Run small PQ jobs and read their output with PQSetup, PQAnalysis, PQEnalyzer and PQViewer."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMATS = Path(__file__).with_name("stack_test_formats.json")
EXAMPLES = ROOT / "examples"

MD_PREFIX = "h2o-md"
MD_STEPS = 100
MD_OUTPUT_FREQ = 10
MD_FRAMES = MD_STEPS // MD_OUTPUT_FREQ
MD_ATOMS = 6000
HESSIAN_PREFIX = "oh-opt"
# liquid water; the start file warms from about 190 K in the first 100 steps
TEMPERATURE_RANGE = (100.0, 500.0)

# logs, timings and the reference list are text, not data
UNCHECKED_SUFFIXES = {".log", ".timings", ".ref"}

NUMBER = re.compile(
    r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|[+-]?(nan|inf)", re.IGNORECASE
)
VERSION = re.compile(r"v\d+\.\d+\.\d+\S*")


def line_shape(line):
    """A line with every number replaced by '#' and every version by 'vX'."""
    tokens = []
    for token in line.split():
        if NUMBER.fullmatch(token):
            tokens.append("#")
        elif VERSION.fullmatch(token):
            tokens.append("vX")
        else:
            tokens.append(token)
    return " ".join(tokens)


def file_shapes(path):
    with open(path, encoding="utf-8", errors="replace") as file:
        return sorted({line_shape(line) for line in file if line.strip()})


def output_formats(run_dir, prefix):
    formats = {}
    for path in sorted(run_dir.glob(f"{prefix}.*")):
        suffix = path.name[len(prefix) :]
        if path.suffix in UNCHECKED_SUFFIXES or not path.is_file():
            continue
        formats[suffix] = file_shapes(path)
    return formats


def compare_formats(expected, actual):
    """Return one line per difference between two {suffix: shapes} maps."""
    differences = []
    for suffix in sorted(set(expected) - set(actual)):
        differences.append(f"{suffix}: file is no longer written")
    for suffix in sorted(set(actual) - set(expected)):
        differences.append(f"{suffix}: new file")
    for suffix in sorted(set(expected) & set(actual)):
        removed = sorted(set(expected[suffix]) - set(actual[suffix]))
        added = sorted(set(actual[suffix]) - set(expected[suffix]))
        differences.extend(f"{suffix}: line gone: {shape[:120]}" for shape in removed)
        differences.extend(f"{suffix}: new line: {shape[:120]}" for shape in added)
    return differences


class StackTest:
    def __init__(self, pq, work):
        self.pq = str(Path(pq).resolve())
        self.work = Path(work).resolve()
        self.md_dir = self.work / "md"
        self.hessian_dir = self.work / "hessian"
        self.results = []

    def check(self, tool, name, function, needs=()):
        failed = [need for need in needs if not self.passed(need)]
        if failed:
            self.results.append(
                (tool, name, None, f"skipped: {', '.join(failed)} failed")
            )
            print(f"[skipped] {tool}: {name}", flush=True)
            return
        try:
            detail = function() or ""
            self.results.append((tool, name, True, detail))
        except Exception as error:  # noqa: BLE001
            detail = f"{type(error).__name__}: {error}"
            self.results.append((tool, name, False, detail))
            traceback.print_exc()
        state = "ok" if self.results[-1][2] else "FAILED"
        print(f"[{state}] {tool}: {name} {self.results[-1][3]}".rstrip(), flush=True)

    def passed(self, name):
        return any(passed for _, check, passed, _ in self.results if check == name)

    def run_pq(self, run_dir, input_file):
        result = subprocess.run(
            [self.pq, input_file],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        (run_dir / f"{input_file}.out").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
            raise RuntimeError(
                f"PQ exited with {result.returncode}: {' | '.join(tail)}"
            )

    def write_md_input(self):
        from pqsetup.input_writer import render_input
        from pqsetup.models import SimulationSetup

        self.md_dir.mkdir(parents=True)
        for name in (
            "guff.dat",
            "moldescriptor.dat",
            "input.rst",
            "shake.top",
            "mshake.top",
        ):
            shutil.copy(EXAMPLES / "h2o_mm-mshake" / name, self.md_dir / name)
        setup = SimulationSetup(
            job_type="mm-md",
            ensemble="NVT",
            runner=None,
            start_file="input.rst",
            file_prefix=MD_PREFIX,
            steps=MD_STEPS,
            timestep_fs=2.0,
            temperature_k=298.15,
            thermostat="velocity_rescaling",
            thermostat_relaxation_ps=0.1,
            initialize_velocities=False,
            mm_force_field="off",
            coulomb_cutoff_angstrom=12.0,
            moldescriptor_file="moldescriptor.dat",
            guff_file="guff.dat",
            topology_file="shake.top",
            mshake_file="mshake.top",
            extra_settings={
                "shake": "mshake",
                "output_freq": MD_OUTPUT_FREQ,
                "cell-list": "on",
            },
        )
        result = render_input(setup)
        errors = [item for item in result.diagnostics if item.severity == "error"]
        if not result.valid:
            raise RuntimeError("; ".join(item.message for item in errors))
        (self.md_dir / "run-01.in").write_text(result.input_text, encoding="utf-8")
        return f"{len(result.input_text.splitlines())} lines"

    def run_md(self):
        self.run_pq(self.md_dir, "run-01.in")

    def run_hessian(self):
        shutil.copytree(EXAMPLES / "oh_mm_hessian", self.hessian_dir)
        self.run_pq(self.hessian_dir, "run-01.in")

    def actual_formats(self):
        return {
            "md": output_formats(self.md_dir, MD_PREFIX),
            "hessian": output_formats(self.hessian_dir, HESSIAN_PREFIX),
        }

    def check_formats(self):
        expected = json.loads(FORMATS.read_text(encoding="utf-8"))
        actual = self.actual_formats()
        differences = []
        for run in sorted(set(expected) | set(actual)):
            differences.extend(
                f"{run}{line}"
                for line in compare_formats(expected.get(run, {}), actual.get(run, {}))
            )
        if differences:
            for line in differences:
                print(f"  {line}")
            raise AssertionError(
                f"{len(differences)} differences, update with --update-formats if intended"
            )
        return f"{sum(len(files) for files in actual.values())} files unchanged"

    def pqanalysis_trajectories(self):
        from PQAnalysis.io import read_trajectory

        for suffix, traj_format in (
            (".xyz", "xyz"),
            (".vel", "vel"),
            (".force", "force"),
            (".chrg", "charge"),
        ):
            trajectory = read_trajectory(
                str(self.md_dir / f"{MD_PREFIX}{suffix}"), traj_format=traj_format
            )
            if len(trajectory) != MD_FRAMES or trajectory[0].n_atoms != MD_ATOMS:
                raise AssertionError(
                    f"{suffix}: {len(trajectory)} frames, {trajectory[0].n_atoms} atoms"
                )
        return f"4 files, {MD_FRAMES} frames each"

    def pqanalysis_restart(self):
        from PQAnalysis.io import read_restart_file

        system = read_restart_file(
            str(self.md_dir / f"{MD_PREFIX}.rst"),
            str(self.md_dir / "moldescriptor.dat"),
        )
        if system.n_atoms != MD_ATOMS:
            raise AssertionError(f"{system.n_atoms} atoms")

    def pqanalysis_energy(self):
        from PQAnalysis.io import EnergyFileReader

        energy = EnergyFileReader(str(self.md_dir / f"{MD_PREFIX}.en")).read()
        temperature = energy.temperature
        low, high = TEMPERATURE_RANGE
        if len(temperature) != MD_FRAMES or not all(
            low < value < high for value in temperature
        ):
            raise AssertionError(
                f"{len(temperature)} rows, T {min(temperature):.1f}-{max(temperature):.1f} K"
            )
        return f"{len(temperature)} rows"

    def pqanalysis_rdf(self):
        input_file = self.md_dir / "O-O.rdf.in"
        input_file.write_text(
            f"reference_selection = O\ntarget_selection = O\ndelta_r = 0.05\n"
            f"out_file = O-O_rdf.out\ntraj_files\n./{MD_PREFIX}.xyz\nEND\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["rdf", input_file.name],
            cwd=self.md_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        values = numeric_rows(self.md_dir / "O-O_rdf.out")
        if not values or not all(
            math.isfinite(value) for row in values for value in row
        ):
            raise AssertionError("empty or non-finite RDF")
        return f"{len(values)} bins"

    def pqanalysis_vibrations(self):
        input_file = self.hessian_dir / "vibrations.in"
        text = input_file.read_text(encoding="utf-8").replace(
            "oh-no-opt.hessian", f"{HESSIAN_PREFIX}.hessian"
        )
        input_file.write_text(text, encoding="utf-8")
        subprocess.run(
            ["vibrations", input_file.name],
            cwd=self.hessian_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        values = numeric_rows(self.hessian_dir / "wavenumbers.dat")
        if not values:
            raise AssertionError("no wavenumbers")
        return f"{len(values)} modes"

    def pqenalyzer_energy(self):
        from PQEnalyzer.readers import create_reader
        from PQEnalyzer.statistics import Statistic

        reader = create_reader([str(self.md_dir / f"{MD_PREFIX}.en")])
        _, mean = Statistic.mean(reader.energies, "TEMPERATURE")
        value = float(mean[0])
        if not TEMPERATURE_RANGE[0] < value < TEMPERATURE_RANGE[1]:
            raise AssertionError(f"mean temperature {value}")
        return f"mean T {value:.1f} K"

    def pqenalyzer_optimizer(self):
        from PQEnalyzer.readers import create_reader

        files = sorted(self.hessian_dir.glob(f"{HESSIAN_PREFIX}*.opt"))
        if not files:
            raise AssertionError("PQ wrote no optimizer file")
        reader = create_reader([str(files[0])])
        if not reader.energies:
            raise AssertionError("no optimizer rows")

    def pqviewer_run(self):
        from pqviewer.sources import open_run_dataset

        dataset = open_run_dataset(self.md_dir / "run-01.in")
        if dataset.frame_count != MD_FRAMES:
            raise AssertionError(f"{dataset.frame_count} frames")
        for index in (0, dataset.frame_count - 1):
            frame = dataset.get_frame(index)
            if frame.positions.shape != (MD_ATOMS, 3):
                raise AssertionError(
                    f"frame {index}: positions {frame.positions.shape}"
                )
            if frame.forces is None or frame.velocities is None:
                raise AssertionError(f"frame {index}: forces or velocities missing")
        dataset.manifest()
        return f"{dataset.frame_count} frames"

    def run(self, update_formats):
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True)
        md, hessian = ("MD run",), ("optimizer + Hessian run",)
        self.check("PQSetup", "write MD input", self.write_md_input)
        self.check("PQ", "MD run", self.run_md, needs=("write MD input",))
        self.check("PQ", "optimizer + Hessian run", self.run_hessian)
        if (
            update_formats
            and self.passed("MD run")
            and self.passed("optimizer + Hessian run")
        ):
            FORMATS.write_text(
                json.dumps(self.actual_formats(), indent=1) + "\n", encoding="utf-8"
            )
            print(f"wrote {FORMATS.relative_to(ROOT)}")
        self.check("PQ", "output format", self.check_formats, needs=md + hessian)
        self.check("PQAnalysis", "trajectories", self.pqanalysis_trajectories, needs=md)
        self.check("PQAnalysis", "restart file", self.pqanalysis_restart, needs=md)
        self.check("PQAnalysis", "energy file", self.pqanalysis_energy, needs=md)
        self.check("PQAnalysis", "rdf CLI", self.pqanalysis_rdf, needs=md)
        self.check(
            "PQAnalysis", "vibrations CLI", self.pqanalysis_vibrations, needs=hessian
        )
        self.check("PQEnalyzer", "energy file", self.pqenalyzer_energy, needs=md)
        self.check(
            "PQEnalyzer", "optimizer file", self.pqenalyzer_optimizer, needs=hessian
        )
        self.check("PQViewer", "open MD run", self.pqviewer_run, needs=md)
        return all(passed for _, _, passed, _ in self.results)

    def summary(self, title):
        lines = [f"### {title}", "", "| Tool | Check | Result |", "|---|---|---|"]
        for tool, name, passed, detail in self.results:
            state = {True: "✅", False: "❌", None: "⏭️"}[passed]
            lines.append(
                f"| {tool} | {name} | {state} {detail.replace('|', '/')[:160]} |"
            )
        return "\n".join(lines) + "\n"


def numeric_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens and all(NUMBER.fullmatch(token) for token in tokens):
            rows.append([float(token) for token in tokens])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", required=True, help="PQ executable")
    parser.add_argument(
        "--work", default="stack-test", help="scratch directory, emptied first"
    )
    parser.add_argument("--title", default="Stack test")
    parser.add_argument(
        "--update-formats",
        action="store_true",
        help=f"rewrite {FORMATS.name} from this run",
    )
    args = parser.parse_args()

    test = StackTest(args.pq, args.work)
    passed = test.run(args.update_formats)
    summary = test.summary(args.title)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as file:
            file.write(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
