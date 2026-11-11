#!/usr/bin/env python3
"""Compare Perl and Python Turbomole drivers on real Turbomole; prints PASS/FAIL, writes results.tar.gz.

Usage: validate_turbomole.py <src/QM/scripts> [--template-ricc2 F] [--template-scf F]
       [--template-ridft F] [--work DIR]
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

BOHR = 1.0 / 0.529177210903
WATER = [("o", 0.0, 0.0, 0.0), ("h", 0.758, 0.0, 0.504), ("h", -0.758, 0.0, 0.504)]
FIELD = [(0.0, 0.0, -2.9, -0.834), (0.757, 0.0, -3.486, 0.417), (-0.757, 0.0, -3.486, 0.417)]
STEPS = [  # (use point charges, run define, geometry shift in A)
    (False, True, 0.00), (False, False, 0.02), (True, True, 0.00), (True, False, 0.02)]
PAIRS = {"ricc2": ("turbomole_ricc2", "turbomole_ricc2.py"),
         "scf": ("turbomole_hf-dft", "turbomole_hf_dft.py"),
         "ridft": ("turbomole_ridft", "turbomole_ridft.py")}
COMPARED = ("qm_forces", "qm_charges")


def scf_template(ricc2_template):
    """Drop the cc2 block of the RI-CC2 define template."""
    lines = ricc2_template.read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == "cc2")
    end = next(i for i, line in enumerate(lines) if line.strip().startswith("geoopt"))
    return "".join(lines[:start] + lines[end + 2:])


def write_inputs(work, shift):
    rows = "".join(f"{x * BOHR:20.14f}  {(y + shift) * BOHR:20.14f}  {z * BOHR:20.14f}      {s}\n"
                   for s, x, y, z in WATER)
    (work / "coord").write_text("$coord\n" + rows + "$end\n")
    (work / "pc").write_text("".join(
        f"{x * BOHR:16.12f}\t{y * BOHR:16.12f}\t{z * BOHR:16.12f}\t{q:16.12f}\n"
        for x, y, z, q in FIELD))


def run_sequence(scripts, script, template, work):
    work.mkdir(parents=True)
    shutil.copyfile(template, work / "tm.template")
    steps = []
    for index, (use_pc, define, shift) in enumerate(STEPS):
        write_inputs(work, shift)
        interpreter = [sys.executable] if script.endswith(".py") else ["perl"]
        done = subprocess.run(
            interpreter + [str(scripts / script), "0", "1" if define else "0",
                           "1" if use_pc else "0", "tm.template", "pc"],
            cwd=work, capture_output=True, text=True, check=False)
        results = {name: (work / name).read_bytes() if (work / name).exists() else None
                   for name in COMPARED}
        keep = work / f"step{index}"
        keep.mkdir()
        for name in COMPARED + ("tm.out", "dscf_nopc.out", "ridft_nopc.out"):
            if (work / name).exists():
                shutil.copyfile(work / name, keep / name)
        steps.append({"returncode": done.returncode, "stderr": done.stderr[-500:], **results})
    return steps


def field_free_energies(nopc_out):
    """Item 3 of #1125: first-iteration line vs '| total' line in the same run."""
    text = nopc_out.read_text(errors="replace") if nopc_out.exists() else ""
    first = re.findall(r"^\s+1\s+(-\d+\.\d+)\s+-\d+.\d+\s+\d+.\d+\s+\d+.\d+D\+\d+", text, re.M)
    total = re.findall(r"^\s+\|\s+total\s+energy\s+=\s+(\S+)", text, re.M)
    return (float(first[-1]) if first else None,
            float(total[-1].replace("D", "E")) if total else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", type=Path)
    here = Path(__file__).resolve().parents[2]
    parser.add_argument("--template-ricc2", type=Path,
                        default=here / "examples/malondialdehyde_tm-rpmd/tm_define.template")
    parser.add_argument("--template-scf", type=Path)
    parser.add_argument("--template-ridft", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    scripts = args.scripts.resolve()
    work = (args.work or Path(tempfile.mkdtemp(prefix="pq-tm-validation-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not shutil.which("dscf"):
        print("Turbomole not found: load it so dscf, grad, ricc2 are on PATH", file=sys.stderr)
        return 1

    templates = {"ricc2": args.template_ricc2.resolve()}
    scf = args.template_scf
    if scf is None:
        scf = work / "scf.template"
        scf.write_text(scf_template(args.template_ricc2))
    templates["scf"] = scf.resolve()
    if args.template_ridft:
        templates["ridft"] = args.template_ridft.resolve()

    report, ok = {"steps": [list(s) for s in STEPS], "cases": {}}, True
    for kind, template in templates.items():
        perl, python = PAIRS[kind]
        old = run_sequence(scripts, perl, template, work / kind / "perl")
        new = run_sequence(scripts, python, template, work / kind / "python")
        for index, (a, b) in enumerate(zip(old, new)):
            same = all(a[n] == b[n] for n in COMPARED) and (a["returncode"] == 0) == (b["returncode"] == 0)
            ran = a["returncode"] == 0 and a["qm_forces"] is not None
            passed = same and ran
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {python:<20} step {index} "
                  f"(pc={int(STEPS[index][0])}, define={int(STEPS[index][1])})"
                  + ("" if ran else f"  Perl failed: {a['stderr'].strip()[:120]}")
                  + ("" if same else "  outputs differ"))
            report["cases"][f"{python} step {index}"] = {
                "pass": passed, "identical": same, "perl_returncode": a["returncode"],
                "python_returncode": b["returncode"]}
    nopc = work / "scf" / "perl" / "step3" / "dscf_nopc.out"
    first, total = field_free_energies(nopc)
    report["item3_field_free_energy"] = {"first_iteration": first, "total_line": total}
    print(f"INFO  field-free energy: first-iteration line {first}, '| total' line {total}")

    tm_out = work / "ricc2" / "perl" / "step0" / "tm.out"
    version = re.search(r"TURBOMOLE.*?(V\d[\w.\-]*)", tm_out.read_text(errors="replace")) \
        if tm_out.exists() else None
    report["turbomole_version"] = version.group(1) if version else None
    (work / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with tarfile.open(work / "results.tar.gz", "w:gz") as archive:
        archive.add(work / "results.json", arcname="results.json")
        for path in sorted(work.glob("*/*/step*/qm_*")):
            archive.add(path, arcname=str(path.relative_to(work)))
    print(f"{'ALL PASS' if ok else 'FAILED'}; attach {work / 'results.tar.gz'} to the PR")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
