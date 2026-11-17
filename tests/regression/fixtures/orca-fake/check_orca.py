#!/usr/bin/env python3
"""Check PQ output against fake_orca.py calls; usage: check_orca.py qm|qmmm."""
import json
import math
from pathlib import Path
import re
import sys

HARTREE_TO_KCAL = 627.5096080305927
BOHR_IN_ANGSTROM = 0.529177210903
GRAD_TO_FORCE = HARTREE_TO_KCAL / BOHR_IN_ANGSTROM
BASE_CHARGES = {"O": -0.66, "H": 0.33, "K": 1.0}
MM_CHARGES = (-0.82, 0.41)
LAYER, SMOOTHING, PC_SHELL = 3.8, 0.2, 15.0

errors = []


def check(condition, message):
    if not condition:
        errors.append(message)
    return condition


def frames(path):
    lines = Path(path).read_text().splitlines()
    result, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        result.append([line.split() for line in lines[i + 2:i + 2 + n]])
        i += 2 + n
    return result


def info_value(prefix, label):
    text = Path(prefix + ".info").read_text()
    match = re.search(re.escape(label) + r"\s+(-?\d+(?:\.\d*)?)", text)
    return float(match.group(1)) if match else None


def expected_charges(symbols, total, scale):
    values = [BASE_CHARGES.get(s, 0.0) * scale for s in symbols]
    values[0] += total - sum(values)
    return values


def keyword_set(call):
    return {k.lower() for k in call["keywords"].split()}


def common(call):
    check(not re.search(r"__[A-Z]+__", call["input"]), f"{call['stem']}: unresolved placeholder")
    check("noautostart" in keyword_set(call), f"{call['stem']}: NoAutoStart missing")


def check_pure(calls, stem, prefix, method, charge_scale):
    mine = [c for c in calls if c["stem"] == stem]
    if not check(len(mine) >= 2, f"{stem}: expected >= 2 ORCA calls, got {len(mine)}"):
        return
    for index, call in enumerate(mine):
        common(call)
        kw = keyword_set(call)
        check({method, "engrad"} <= kw and "noiter" not in kw,
              f"{stem}: unexpected keywords '{call['keywords']}'")
        check(call["pointcharges"] is None, f"{stem}: point charges in pure QM")
        if index == 0:
            check(call["moinp"] is None, f"{stem}: first step must not reuse orbitals")
        else:
            check(call["moinp"] is not None and call["moinp"].endswith("orca_previous.gbw"),
                  f"{stem}: step {index} does not reuse orca_previous.gbw")

    xyz, force, chrg = (frames(prefix + s) for s in (".xyz", ".force", ".chrg"))
    matched = 0
    for frame, forces, charges in zip(xyz, force, chrg):
        positions = [[float(x) for x in atom[1:4]] for atom in frame]
        call = next((c for c in mine if all(
            abs(a - b) < 1e-6 for p, q in zip(positions, c["positions_angstrom"])
            for a, b in zip(p, q))), None)
        if call is None:
            continue
        matched += 1
        for atom, grad in zip(forces, call["gradient"]):
            for value, g in zip(atom[1:4], grad):
                check(abs(float(value) + g * GRAD_TO_FORCE) < 1e-5,
                      f"{prefix}: force {value} != {-g * GRAD_TO_FORCE:.8f} kcal/mol/A")
        want = expected_charges(call["symbols"], call["charge"], charge_scale)
        for atom, q in zip(charges, want):
            check(abs(float(atom[1]) - q) < 1e-6,
                  f"{prefix}: charge {atom[1]} != {q:.8f} (wrong population table?)")
    check(matched > 0, f"{prefix}: no output frame matches an ORCA geometry")

    e_qm = info_value(prefix, "E(QM)")
    check(e_qm is not None and abs(e_qm - mine[-1]["energy"] * HARTREE_TO_KCAL) < 1e-3,
          f"{prefix}: E(QM) {e_qm} != {mine[-1]['energy'] * HARTREE_TO_KCAL:.5f} kcal/mol")


def check_point_charges(call):
    rows = call["pc_rows"]
    if not check(rows and len(rows) == call["npc"], f"{call['stem']}: bad pc count"):
        return
    qm = call["positions_angstrom"]
    centre = qm[call["symbols"].index("K")]
    distances = []
    for q, *pos in rows:
        check(any(abs(q - m) < 1e-6 for m in MM_CHARGES), f"point charge {q} is not an MM charge")
        distances.append(math.dist(pos, centre))
        check(min(math.dist(pos, a) for a in qm) > 0.3, "point charge sits on a QM atom")
    # 1.5 A: atom-to-molecule-centre slack
    check(min(distances) > LAYER - SMOOTHING - 1.5,
          f"point charge at {min(distances):.2f} A, inside the QM layer")
    check(max(distances) < LAYER + PC_SHELL + 1.5,
          f"point charge at {max(distances):.2f} A, outside the pc shell (Bohr instead of A?)")
    check(max(distances) > PC_SHELL, f"pc shell only reaches {max(distances):.2f} A")


def check_qmmm(calls, prefix, stages):
    names = [s[0] for s in stages]
    mine = [c for c in calls if c["stem"] in names]
    if not check(mine and len(mine) % len(stages) == 0,
                 f"{prefix}: ORCA calls {[c['stem'] for c in mine]} do not repeat {names}"):
        return
    for start in range(0, len(mine), len(stages)):
        polarized = mine[start]
        for call, (stem, need, avoid) in zip(mine[start:start + len(stages)], stages):
            check(call["stem"] == stem, f"{prefix}: expected {stem}, got {call['stem']}")
            common(call)
            kw = keyword_set(call)
            check(need <= kw and not (avoid & kw), f"{stem}: unexpected keywords '{call['keywords']}'")
            check(call["charge"] == 1 and "K" in call["symbols"], f"{stem}: QM charge {call['charge']} != 1")
            if call is polarized:
                check(call["pointcharges"] is not None, f"{stem}: no point charges")
                check_point_charges(call)
            else:
                check(call["pointcharges"] is None, f"{stem}: point charges in field-free stage")
                check(call["moinp"] is not None and call["moinp"].endswith(polarized["stem"] + ".gbw"),
                      f"{stem}: does not read {polarized['stem']}.gbw")
                check("maxiter 0" in call["input"].lower(), f"{stem}: SCF not frozen")
                check(call["positions_angstrom"] == polarized["positions_angstrom"],
                      f"{stem}: geometry differs from the polarized stage")
    n_qm = info_value(prefix, "N(QM-ATOMS)")
    check(n_qm == len(mine[-1]["symbols"]), f"{prefix}: N(QM-ATOMS) {n_qm} != {len(mine[-1]['symbols'])}")


def main(mode):
    calls = [json.loads(line) for line in Path("fake_orca_calls.jsonl").read_text().splitlines()]
    if mode == "qm":
        check_pure(calls, "orca_hf", "water-hf", "hf", 1.0)
        check_pure(calls, "orca_dft", "water-dft", "dft", 1.0)
        check_pure(calls, "orca_mp2", "water-rimp2", "ri-mp2", 0.9)
    else:
        frozen = {"noiter", "calcguessenergy"}
        check_qmmm(calls, "qmmm-dft", [
            ("orca_dft_polarized", {"dft"}, {"noiter", "engrad"}),
            ("orca_dft_frozen", frozen | {"dft", "engrad"}, set()),
        ])
        check_qmmm(calls, "qmmm-rimp2", [
            ("orca_polarized", {"hf"}, {"noiter", "engrad"}),
            ("orca_frozen_hf", frozen | {"hf"}, {"engrad"}),
            ("orca_mp2", {"ri-mp2", "noiter", "engrad"}, set()),
        ])
    for message in dict.fromkeys(errors):
        print("FAIL:", message, file=sys.stderr)
    print(f"{mode}: {len(calls)} ORCA calls checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
