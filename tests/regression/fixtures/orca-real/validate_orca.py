#!/usr/bin/env python3
"""Validate PQ's ORCA drivers against a real ORCA installation.

Usage: validate_orca.py <scripts dir> <examples/orca dir> [work dir]
ORCA_EXE (or orca on PATH) selects ORCA. Writes results.json, exits 1 on FAIL.

Checks per driver (orca_hf.py, orca_dft.py, orca_rimp2.py) on water:
  1. pure QM gradient == central finite differences of the driver energy
  2. QM/MM with all point charges zero == pure QM (energy, gradient)
  3. HF/DFT: polarized-then-frozen energy >= pure QM energy (variational)
  4. a real point-charge field changes the returned atomic charges
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

WATER = [("O", 0.000, 0.0, 0.000), ("H", 0.758, 0.0, 0.504), ("H", -0.758, 0.0, 0.504)]
# One TIP3P-like water 3 A from the QM oxygen; charges in e, positions in A.
FIELD = [(-0.834, 0.0, 0.0, -2.900), (0.417, 0.757, 0.0, -3.486), (0.417, -0.757, 0.0, -3.486)]
STEP = 0.005  # finite-difference step in A
BOHR_IN_ANGSTROM = 0.529177210903
DRIVERS = {
    "orca_hf.py": ("orca_hf.template", 2e-6),
    "orca_dft.py": ("orca_dft.template", 1e-4),   # DFT grid noise
    "orca_rimp2.py": ("orca.template", 2e-6),
}


def run(scripts, template, driver, atoms, charges=None, work=None):
    work.mkdir(parents=True, exist_ok=True)
    (work / "orca_coord.xyz").write_text(
        f"{len(atoms)}\n\n" + "".join(f"{s} {x:.12f} {y:.12f} {z:.12f}\n" for s, x, y, z in atoms))
    pc = work / "pc"
    if charges:
        pc.write_text(f"{len(charges)}\n" + "".join(
            f"{q:.12f} {x:.12f} {y:.12f} {z:.12f}\n" for q, x, y, z in charges))
    done = subprocess.run(
        [sys.executable, str(scripts / driver), "0", "1", "1" if charges else "0",
         str(template), str(pc)],
        cwd=work, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"{driver} failed in {work}: {done.stderr.strip()}")
    lines = (work / "qm_forces").read_text().split("\n")
    energy = float(lines[0])
    gradient = [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
    q = [float(v) for v in (work / "qm_charges").read_text().split()]
    return energy, gradient, q


def validate(scripts, examples, work, driver):
    template_name, fd_tol = DRIVERS[driver]
    base = work / driver.removesuffix(".py")
    base.mkdir(parents=True, exist_ok=True)
    # TightSCF noise in the non-variational MP2 energy alone gives ~3e-6 Eh/bohr.
    template = base / template_name
    template.write_text((examples / template_name).read_text().replace("TightSCF", "VeryTightSCF"))
    results = []

    def record(name, ok, detail):
        results.append({"driver": driver, "check": name, "pass": bool(ok), "detail": detail})

    e0, g0, q0 = run(scripts, template, driver, WATER, work=base / "pure")

    worst = 0.0
    for i in range(len(WATER)):
        for d in range(3):
            e = {}
            for k in (2, 1, -1, -2):
                atoms = [list(a) for a in WATER]
                atoms[i][d + 1] += k * STEP
                e[k] = run(scripts, template, driver, atoms, work=base / f"fd_{i}{d}_{k}")[0]
            # Five-point stencil: truncation error O(h^4), central differences O(h^2) are ~2e-5 here.
            fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * STEP / BOHR_IN_ANGSTROM)
            worst = max(worst, abs(fd - g0[i][d]))
    record("gradient_vs_finite_difference", worst < fd_tol,
           f"max |dE/dx - gradient| = {worst:.2e} Eh/bohr (tol {fd_tol:.0e})")

    zero = [(0.0, x, y, z) for _, x, y, z in FIELD]
    ez, gz, _ = run(scripts, template, driver, WATER, zero, work=base / "zero_field")
    dg = max(abs(a - b) for ra, rb in zip(gz, g0) for a, b in zip(ra, rb))
    record("zero_charges_equal_pure_qm", abs(ez - e0) < 1e-6 and dg < 1e-5,
           f"dE = {ez - e0:.2e} Eh, max dgrad = {dg:.2e} Eh/bohr")

    ef, _, qf = run(scripts, template, driver, WATER, FIELD, work=base / "field")
    if driver != "orca_rimp2.py":  # MP2 is not variational in the orbitals
        record("frozen_energy_above_pure_qm", ef - e0 > -1e-8,
               f"E(frozen) - E(pure) = {ef - e0:.3e} Eh")
    dq = max(abs(a - b) for a, b in zip(qf, q0))
    record("field_polarizes_charges", dq > 1e-3 and abs(sum(qf)) < 1e-4,
           f"max |dq| = {dq:.4f} e, sum q = {sum(qf):.1e} e")
    return results, e0


def main():
    scripts, examples = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not shutil.which(os.environ.get("ORCA_EXE", "orca")):
        print("ORCA not found: set ORCA_EXE", file=sys.stderr)
        return 1
    work = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(tempfile.mkdtemp(prefix="pq-orca-"))
    results, energies = [], {}
    for driver in DRIVERS:
        try:
            part, energies[driver] = validate(scripts, examples, work, driver)
            results += part
        except RuntimeError as error:
            results.append({"driver": driver, "check": "runs", "pass": False, "detail": str(error)})
    if {"orca_hf.py", "orca_rimp2.py"} <= energies.keys():
        corr = energies["orca_rimp2.py"] - energies["orca_hf.py"]
        results.append({"driver": "orca_rimp2.py", "check": "correlation_energy_negative",
                        "pass": -1.0 < corr < 0.0, "detail": f"E(MP2) - E(HF) = {corr:.6f} Eh"})
    version = subprocess.run([shutil.which(os.environ.get("ORCA_EXE", "orca")), "--version"],
                             capture_output=True, text=True, check=False).stdout.strip()
    (work / "results.json").write_text(json.dumps(
        {"orca_version": version, "results": results}, indent=2) + "\n")
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['driver']:<14} {r['check']:<32} {r['detail']}")
    print(f"results: {work / 'results.json'}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
