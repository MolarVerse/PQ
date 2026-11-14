#!/usr/bin/env python3
"""Compare Perl and Python Turbomole drivers on real Turbomole; prints PASS/FAIL, writes results.tar.gz.

The Perl driver runs real Turbomole; every program call is recorded (input files,
output, files written). The Python driver then runs against a replay of exactly
those calls, so both parse the same Turbomole output and must write byte-identical
results. Usage: validate_turbomole.py <src/QM/scripts> [--template-ricc2 F]
[--template-scf F] [--template-ridft F] [--work DIR]
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
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
PROGRAMS = ("define", "dscf", "grad", "ridft", "rdgrad", "ricc2")
COMPARED = ("qm_forces", "qm_charges")
KEPT = COMPARED + ("tm.out", "dscf_nopc.out", "ridft_nopc.out", "gradient", "energy", "control")

SHIM = r'''#!/usr/bin/env python3
import json, os, shutil, subprocess, sys
from pathlib import Path
name, mode, store = Path(sys.argv[0]).name, os.environ["TM_SHIM_MODE"], Path(os.environ["TM_SHIM_STORE"])
counter = store / "counter"
index = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(index + 1))
call = store / f"{index:03d}"
stdin = sys.stdin.buffer.read() if name == "define" else b""


LOGS = {"tm.out", "dscf_nopc.out", "ridft_nopc.out"}  # written by the driver's tee


def snapshot():
    return {p.name: p.read_bytes() for p in sorted(Path.cwd().iterdir())
            if p.is_file() and p.name not in LOGS}


if mode == "record":
    call.mkdir()
    before = snapshot()
    real = os.environ["TM_SHIM_PATH_" + name]
    done = subprocess.run([real], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    after = snapshot()
    (call / "before").mkdir(); (call / "after").mkdir()
    for key, value in before.items(): (call / "before" / key).write_bytes(value)
    for key, value in after.items(): (call / "after" / key).write_bytes(value)
    (call / "meta.json").write_text(json.dumps(
        {"name": name, "returncode": done.returncode, "stdin": stdin.decode("latin-1")}))
    (call / "output").write_bytes(done.stdout)
    sys.stdout.buffer.write(done.stdout)
    sys.exit(done.returncode)

meta = json.loads((call / "meta.json").read_text())
problems = []
if meta["name"] != name:
    problems.append(f"call {index}: expected {meta['name']}, got {name}")
if meta["stdin"] != stdin.decode("latin-1"):
    problems.append(f"call {index} ({name}): different stdin")
current = snapshot()
recorded = {p.name: p.read_bytes() for p in (call / "before").iterdir()}
for key in sorted(set(current) | set(recorded)):
    if current.get(key) != recorded.get(key):
        problems.append(f"call {index} ({name}): input file {key} differs")
with (store / "problems").open("a") as log:
    for line in problems:
        log.write(line + "\n")
after = {p.name: p.read_bytes() for p in (call / "after").iterdir()}
for key in recorded:
    if key not in after:
        Path(key).unlink(missing_ok=True)
for key, value in after.items():
    if recorded.get(key) != value:
        Path(key).write_bytes(value)
sys.stdout.buffer.write((call / "output").read_bytes())
sys.exit(meta["returncode"])
'''


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


def make_shims(root):
    shims = root / "shims"
    shims.mkdir(parents=True, exist_ok=True)
    env = {}
    for name in PROGRAMS:
        real = shutil.which(name)
        if real:
            env["TM_SHIM_PATH_" + name] = real
        path = shims / name
        path.write_text(SHIM.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return shims, env


def run_sequence(scripts, script, template, work, env):
    work.mkdir(parents=True)
    shutil.copyfile(template, work / "tm.template")
    steps = []
    for index, (use_pc, define, shift) in enumerate(STEPS):
        write_inputs(work, shift)
        interpreter = [sys.executable] if script.endswith(".py") else ["perl"]
        done = subprocess.run(
            interpreter + [str(scripts / script), "0", "1" if define else "0",
                           "1" if use_pc else "0", "tm.template", "pc"],
            cwd=work, env=env, capture_output=True, text=True, check=False)
        results = {name: (work / name).read_bytes() if (work / name).exists() else None
                   for name in COMPARED}
        keep = work / f"step{index}"
        keep.mkdir()
        for name in KEPT:
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
    shims, shim_env = make_shims(work)
    base_env = dict(os.environ, **shim_env, PATH=f"{shims}{os.pathsep}{os.environ['PATH']}")

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
        store = work / kind / "calls"
        store.mkdir(parents=True)
        old = run_sequence(scripts, perl, template, work / kind / "perl",
                           dict(base_env, TM_SHIM_MODE="record", TM_SHIM_STORE=str(store)))
        (store / "counter").unlink()
        new = run_sequence(scripts, python, template, work / kind / "python",
                           dict(base_env, TM_SHIM_MODE="replay", TM_SHIM_STORE=str(store)))
        problems = (store / "problems").read_text().splitlines() if (store / "problems").exists() else []
        for line in problems[:10]:
            print(f"FAIL  {python:<20} {line}")
        ok &= not problems
        for index, (a, b) in enumerate(zip(old, new)):
            same = all(a[n] == b[n] for n in COMPARED) and a["returncode"] == b["returncode"]
            ran = a["returncode"] == 0 and bool(a["qm_forces"])
            charges = bool(a["qm_charges"] and a["qm_charges"].strip())
            passed = same and ran
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {python:<20} step {index} "
                  f"(pc={int(STEPS[index][0])}, define={int(STEPS[index][1])})"
                  + ("" if ran else f"  Perl failed: {a['stderr'].strip()[:120]}")
                  + ("" if same else "  outputs differ")
                  + ("" if charges else "  WARNING: qm_charges is empty"))
            report["cases"][f"{python} step {index}"] = {
                "pass": passed, "identical": same, "charges_written": charges,
                "perl_returncode": a["returncode"], "python_returncode": b["returncode"]}
        report["cases"][f"{python} replay problems"] = problems
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
        for path in sorted(work.glob("*/*/step*/*")):
            archive.add(path, arcname=str(path.relative_to(work)))
    print(f"{'ALL PASS' if ok else 'FAILED'}; attach {work / 'results.tar.gz'} to the PR")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
