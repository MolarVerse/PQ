#!/usr/bin/env python3
"""PQ/ORCA RI-MP2 interface (Python standard library only).

Arguments: charge reset_orbitals use_pointcharges template pointcharge_file.
ORCA_EXE selects the ORCA executable. All calculation files belong to the current
working directory. See examples/orca/README.md for the template contract.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

from orca_common import (
    RESULTS, charges, coordinates, energy, gradient, number,
    run_job, validate_template,
)

def main(argv):
    # Remove results even when called outside PQ, before any validation fails.
    for path in RESULTS:
        path.unlink(missing_ok=True)
    if len(argv) != 5:
        raise ValueError("Usage: orca_rimp2.py charge reset use_pc template pc_file")
    charge = int(argv[0])
    if argv[1] not in ("0", "1") or argv[2] not in ("0", "1"):
        raise ValueError("reset and use_pc must be 0 or 1")
    reset, use_pc = argv[1] == "1", argv[2] == "1"
    template = Path(argv[3]).read_text()
    validate_template(template)
    symbols = coordinates()
    pc = Path(argv[4]).resolve()
    if use_pc:
        rows = pc.read_text().splitlines()
        if int(rows[0]) <= 0 or len(rows) != int(rows[0]) + 1:
            raise ValueError("Invalid ORCA point-charge count")
        for row in rows[1:]:
            values = row.split()
            if len(values) != 4:
                raise ValueError("Point charges require q x y z in e/Angstrom")
            for value in values:
                number(value)
    executable = shutil.which(os.environ.get("ORCA_EXE", "orca"))
    if not executable:
        raise ValueError("ORCA not found; load ORCA or set ORCA_EXE to its full path")
    executable = str(Path(executable).resolve())
    fingerprint = hashlib.sha256(json.dumps(
        [template, symbols, charge, use_pc, executable]).encode()).hexdigest()
    cache = Path("orca_previous.gbw")
    metadata = Path("orca_previous.json")
    guess = None
    if not reset and cache.is_file() and metadata.is_file():
        if json.loads(metadata.read_text()).get("fingerprint") == fingerprint:
            guess = cache.resolve()
    # Invalidate metadata until every stage and all result parsing succeed.
    metadata.unlink(missing_ok=True)
    if use_pc:
        run_job(executable, template, "orca_polarized", "HF", charge, guess, pc)
        reference = Path("orca_polarized.gbw").resolve()
        hf = run_job(executable, template, "orca_frozen_hf",
                     "HF NoIter CalcGuessEnergy", charge, reference, frozen=True)
        hf_energy = energy(hf, "FINAL SINGLE POINT ENERGY")
        mp2 = run_job(executable, template, "orca_mp2", "RI-MP2 NoIter EnGrad",
                      charge, reference, frozen=True)
        total = hf_energy + energy(mp2, r"RI-MP2 CORRELATION ENERGY")
        _, gradients = gradient(Path("orca_mp2.engrad"), symbols)
    else:
        mp2 = run_job(executable, template, "orca_mp2", "RI-MP2 EnGrad", charge, guess)
        reference = Path("orca_mp2.gbw").resolve()
        total, gradients = gradient(Path("orca_mp2.engrad"), symbols)
        if abs(total - energy(mp2, "FINAL SINGLE POINT ENERGY")) > 1e-7:
            raise ValueError("ORCA output and gradient energy disagree")
    atomic_charges = charges(mp2, symbols, charge)
    # Publish results only after every parse succeeds. PQ reads gradients and
    # performs the force sign change and unit conversion itself.
    force_text = f"{total:.16e}\n" + ''.join(
        ' '.join(f"{v:.16e}" for v in row) + '\n' for row in gradients)
    charge_text = ''.join(f"{v:.16e}\n" for v in atomic_charges)
    shutil.copyfile(reference, cache)
    metadata.write_text(json.dumps({"fingerprint": fingerprint}) + '\n')
    for path, contents in zip(RESULTS, (force_text, charge_text)):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(contents)
        temporary.replace(path)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, RuntimeError, IndexError, KeyError) as error:
        for result_file in RESULTS:
            result_file.unlink(missing_ok=True)
        Path("orca_previous.json").unlink(missing_ok=True)
        print(f"PQ ORCA: {error}", file=sys.stderr)
        sys.exit(1)
