#!/usr/bin/env python3
"""Deterministic stand-in for the ORCA executable used by PQ's ORCA drivers.

Energy model, in Hartree and Bohr, with r_c the mean QM position:
    E = E0 + K/2 * sum_i |r_i - r_c|^2 (+ PENALTY for frozen polarized orbitals)
Each call is logged to fake_orca_calls.jsonl for check_orca.py.
"""
import json
from pathlib import Path
import re
import sys

E0 = -76.0
K = 0.02
CORRELATION = -0.2
PENALTY = 1.0e-3
BOHR_IN_ANGSTROM = 0.529177210903
BASE_CHARGES = {"O": -0.66, "H": 0.33, "K": 1.0}
ATOMIC_NUMBERS = {"H": 1, "O": 8, "K": 19}


def charges(symbols, total, scale):
    values = [BASE_CHARGES.get(s, 0.0) * scale for s in symbols]
    values[0] += total - sum(values)
    return values


def mulliken(symbols, values):
    rows = "".join(f"{i:4d} {s:<2}:{q:16.10f}\n"
                   for i, (s, q) in enumerate(zip(symbols, values)))
    return "MULLIKEN ATOMIC CHARGES\n-----------------------\n" + rows + \
        f"Sum of atomic charges:{sum(values):16.10f}\n\n"


def main():
    inp = Path(sys.argv[1])
    stem = inp.stem
    text = inp.read_text()
    keywords = " ".join(line[1:].strip() for line in text.splitlines()
                        if line.lstrip().startswith("!"))
    pc = re.search(r'%pointcharges\s+"([^"]+)"', text)
    moinp = re.search(r'%moinp\s+"([^"]+)"', text)
    coords = re.search(r"\*\s+xyzfile\s+(-?\d+)\s+(\d+)\s+(\S+)", text)
    charge = int(coords.group(1))

    lines = Path(coords.group(3)).read_text().splitlines()
    atoms = [line.split() for line in lines[2:2 + int(lines[0])]]
    symbols = [a[0] for a in atoms]
    positions = [[float(x) / BOHR_IN_ANGSTROM for x in a[1:4]] for a in atoms]
    center = [sum(p[d] for p in positions) / len(positions) for d in range(3)]
    deltas = [[p[d] - center[d] for d in range(3)] for p in positions]

    noiter = re.search(r"\bNoIter\b", keywords, re.I) is not None
    mp2 = re.search(r"\bRI-MP2\b", keywords, re.I) is not None
    engrad = re.search(r"\bEnGrad\b", keywords, re.I) is not None
    polarized_guess = bool(moinp) and "polarized" in Path(moinp.group(1)).name

    reference = E0 + 0.5 * K * sum(x * x for d in deltas for x in d)
    if noiter and polarized_guess:
        reference += PENALTY
    total = reference + (CORRELATION if mp2 else 0.0)
    gradient = [[K * x for x in d] for d in deltas]

    npc, pc_rows = 0, []
    if pc:
        pc_lines = Path(pc.group(1)).read_text().splitlines()
        npc = int(pc_lines[0])
        pc_rows = [[float(x) for x in row.split()] for row in pc_lines[1:]]

    out = ["                 * O   R   C   A *  (fake for PQ tests)\n"]
    if not noiter:
        out.append("SCF CONVERGED AFTER   7 CYCLES\n")
    out.append(f"Total Energy       :{reference:22.12f} Eh\n")
    out.append(mulliken(symbols, charges(symbols, charge, 1.0)))
    if mp2:
        out.append(f"RI-MP2 CORRELATION ENERGY:{CORRELATION:22.12f} Eh\n")
        out.append("RELAXED MP2 DENSITY POPULATION ANALYSIS\n")
        out.append(mulliken(symbols, charges(symbols, charge, 0.9)))
    out.append(f"FINAL SINGLE POINT ENERGY {total:22.12f}\n")
    out.append("                             ****ORCA TERMINATED NORMALLY****\n")
    sys.stdout.write("".join(out))

    Path(stem + ".gbw").write_text(json.dumps({"stem": stem, "npc": npc}))
    if engrad:
        rows = [f"{len(symbols)}", f"{total:.15e}"]
        rows += [f"{g:.15e}" for row in gradient for g in row]
        rows += [f"{ATOMIC_NUMBERS.get(s, 0)} " + " ".join(f"{x:.10f}" for x in p)
                 for s, p in zip(symbols, positions)]
        Path(stem + ".engrad").write_text(
            "#\n# fake ORCA engrad\n#\n" + "\n".join(rows) + "\n")

    with Path("fake_orca_calls.jsonl").open("a") as log:
        log.write(json.dumps({
            "stem": stem,
            "input": text,
            "keywords": keywords,
            "charge": charge,
            "pointcharges": pc.group(1) if pc else None,
            "npc": npc,
            "pc_rows": pc_rows,
            "moinp": moinp.group(1) if moinp else None,
            "symbols": symbols,
            "positions_angstrom": [[float(x) for x in a[1:4]] for a in atoms],
            "energy": total,
            "gradient": gradient if engrad else None,
        }) + "\n")


if __name__ == "__main__":
    main()
