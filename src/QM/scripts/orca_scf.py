"""HF/DFT protocol used by orca_hf.py and orca_dft.py.

With embedding: converge in the field, then evaluate the energy and analytic
QM gradient without the field, preserving the polarized reference orbitals.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from orca_common import (
    NUMBER, RESULTS, coordinates, energy, gradient, number,
    run_job, validate_template,
)


def scf_charges(text, symbols, charge):
    """Read SCF Mulliken charges, including unrestricted charge/spin tables."""
    start = text.rfind("MULLIKEN ATOMIC CHARGES")
    if start < 0:
        raise ValueError("Missing SCF Mulliken atomic charges")
    rows = []
    for line in text[start:].splitlines()[1:]:
        match = re.match(r"\s*(\d+)\s+([A-Za-z]+)\s*:\s*(" + NUMBER + r")", line)
        if match:
            i, symbol, value = match.groups()
            if int(i) != len(rows) or symbol.lower() != symbols[len(rows)].lower():
                raise ValueError("ORCA atomic charge order does not match QM atoms")
            rows.append(number(value))
            if len(rows) == len(symbols):
                break
        elif rows:
            break
    if len(rows) != len(symbols):
        raise ValueError("Incomplete SCF Mulliken atomic charges")
    if abs(sum(rows) - charge) > max(1e-4, len(rows) * 1e-6):
        raise ValueError("SCF Mulliken charges do not sum to the QM charge")
    return rows


def validate_scf_template(template, mode):
    validate_template(template)
    common = '\n'.join(line.split('#', 1)[0] for line in template.splitlines())
    if re.search(r"%\s*(?:mp2|mdci|casscf|tddft|cis|autoci|rocis|mrci)\b|"
                 r"\b(?:[A-Z0-9-]*MP2[A-Z0-9-]*|[A-Z0-9-]*CCSD[A-Z0-9-]*|"
                 r"B2PLYP|B2GP-PLYP|DSD-[A-Z0-9-]+|PWPB95)\b", common, re.I):
        raise ValueError("HF/DFT drivers require a ground-state SCF template, without post-SCF methods")
    wrong_reference = r"\b(?:RKS|UKS|ROKS|DFT)\b" if mode == "HF" else r"\b(?:RHF|UHF|ROHF|HF)\b"
    simple_input = '\n'.join(line for line in common.splitlines()
                             if line.lstrip().startswith('!'))
    if re.search(wrong_reference, simple_input, re.I):
        raise ValueError(f"Template reference does not match the {mode} driver")


def check_scf_output(text, mode):
    if mode == "HF" and re.search(r"^\s*Density Functional\s+Method\s+.*DFT", text, re.M):
        raise ValueError("The HF template selected DFT; use orca_dft.py")
    if mode == "DFT" and re.search(r"^\s*Ab initio Hamiltonian\s+Method\s+.*Hartree-Fock", text, re.M):
        raise ValueError("The DFT template selected HF; use orca_hf.py")
    # Also reject double hybrids selected by a functional name not recognized
    # by template validation. Their post-SCF energy/density is not this protocol.
    if re.search(r"\b(?:RI-)?MP2\s+(?:CORRELATION|TOTAL)\s+ENERGY|"
                 r"\bDOUBLE[- ]HYBRID\s+(?:DFT|CALCULATION)|"
                 r"\bE\(MP2\)", text, re.I):
        raise ValueError("Post-SCF correlation detected; HF/DFT drivers do not support double hybrids")


def main(mode, argv):
    for path in RESULTS:
        path.unlink(missing_ok=True)
    if len(argv) != 5:
        raise ValueError(f"Usage: orca_{mode.lower()}.py charge reset use_pc template pc_file")
    charge = int(argv[0])
    if argv[1] not in ("0", "1") or argv[2] not in ("0", "1"):
        raise ValueError("reset and use_pc must be 0 or 1")
    reset, use_pc = argv[1] == "1", argv[2] == "1"
    template = Path(argv[3]).read_text()
    validate_scf_template(template, mode)
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
        [mode, template, symbols, charge, use_pc, executable]).encode()).hexdigest()
    cache, metadata = Path("orca_previous.gbw"), Path("orca_previous.json")
    guess = None
    if not reset and cache.is_file() and metadata.is_file():
        if json.loads(metadata.read_text()).get("fingerprint") == fingerprint:
            guess = cache.resolve()
    metadata.unlink(missing_ok=True)
    stem = "orca_" + mode.lower()
    if use_pc:
        polarized = stem + "_polarized"
        text = run_job(executable, template, polarized, mode, charge, guess, pc)
        check_scf_output(text, mode)
        reference = Path(polarized + ".gbw").resolve()
        stem += "_frozen"
        text = run_job(executable, template, stem,
                       mode + " NoIter CalcGuessEnergy EnGrad", charge,
                       reference, frozen=True)
    else:
        text = run_job(executable, template, stem, mode + " EnGrad", charge, guess)
        reference = Path(stem + ".gbw").resolve()
    check_scf_output(text, mode)
    total = energy(text, "FINAL SINGLE POINT ENERGY")
    grad_energy, gradients = gradient(Path(stem + ".engrad"), symbols)
    if abs(total - grad_energy) > 1e-7:
        raise ValueError("ORCA output and gradient energy disagree")
    atomic_charges = scf_charges(text, symbols, charge)
    force_text = f"{total:.16e}\n" + ''.join(
        ' '.join(f"{v:.16e}" for v in row) + '\n' for row in gradients)
    charge_text = ''.join(f"{v:.16e}\n" for v in atomic_charges)
    for path, contents in zip(RESULTS, (force_text, charge_text)):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(contents)
        temporary.replace(path)
    shutil.copyfile(reference, cache)
    metadata.write_text(json.dumps({"fingerprint": fingerprint}) + '\n')


def cli(mode):
    try:
        main(mode, sys.argv[1:])
    except (OSError, ValueError, RuntimeError, IndexError, KeyError) as error:
        for result in RESULTS:
            result.unlink(missing_ok=True)
        Path("orca_previous.json").unlink(missing_ok=True)
        print(f"PQ ORCA {mode}: {error}", file=sys.stderr)
        sys.exit(1)
