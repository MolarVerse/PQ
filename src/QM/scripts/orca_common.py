"""Shared file handling for PQ's ORCA drivers (Python standard library)."""

import math
from pathlib import Path
import re
import subprocess

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
TOKENS = ("__METHOD__", "__SCF__", "__GUESS__", "__POINTCHARGES__", "__CHARGE__")
RESULTS = (Path("qm_forces"), Path("qm_charges"))


def number(value):
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("Non-finite ORCA result")
    return result


def energy(text, label):
    matches = re.findall(label + r"\s*(?:[:=])?\s*(" + NUMBER + r")", text)
    if not matches:
        raise ValueError(f"Missing {label} in ORCA output")
    return number(matches[-1])


def gradient(path, symbols):
    values = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            values.extend(line.split())
    n = len(symbols)
    if len(values) != 2 + 7 * n or int(values[0]) != n:
        raise ValueError(f"Incomplete or mismatched ORCA gradient: {path}")
    e = number(values[1])
    g = [number(x) for x in values[2:2 + 3 * n]]
    return e, [g[i:i + 3] for i in range(0, len(g), 3)]


def charges(text, symbols, charge):
    # Select exactly the relaxed MP2 population analysis, never the SCF or
    # unrelaxed tables that precede it.
    headings = list(re.finditer(
        r"\b(?:RELAXED\s+MP2\s+DENSITY|MP2\s+RELAXED\s+DENSITY)", text))
    if not headings:
        raise ValueError("Missing relaxed MP2 density population analysis")
    section = text[headings[-1].end():]
    start = section.find("MULLIKEN ATOMIC CHARGES")
    if start < 0:
        raise ValueError("Missing relaxed MP2 Mulliken charges")
    section = section[start:]
    rows = []
    for line in section.splitlines()[1:]:
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
        raise ValueError("Incomplete relaxed MP2 atomic charges")
    if abs(sum(rows) - charge) > max(1e-4, len(rows) * 1e-6):
        raise ValueError("Relaxed MP2 atomic charges do not sum to the QM charge")
    return rows


def quoted(path):
    value = str(path)
    if any(c in value for c in ('"', '\n', '\r')):
        raise ValueError("ORCA file paths cannot contain quotes or newlines")
    return '"' + value + '"'


def coordinates():
    lines = Path("orca_coord.xyz").read_text().splitlines()
    n = int(lines[0])
    if n <= 0 or len(lines) != n + 2:
        raise ValueError("Invalid orca_coord.xyz")
    symbols = []
    for line in lines[2:]:
        symbol, x, y, z = line.split()
        for value in (x, y, z):
            number(value)
        symbols.append(symbol)
    return symbols


def validate_template(template):
    for token in TOKENS:
        if token not in template:
            raise ValueError(f"ORCA template is missing {token}")
    # The driver owns all stage-specific settings. Keeping these out of the
    # common template prevents accidental reconvergence or field inheritance.
    common = '\n'.join(line.split('#', 1)[0] for line in template.splitlines())
    if re.search(r"\$(?:new_job|newjob)|%\s*(?:scf|pointcharges|moinp|base)\b", common, re.I):
        raise ValueError("ORCA template contains a driver-controlled block")
    if re.search(r"\b(?:NoIter|CalcGuessEnergy|MORead|EnGrad|Opt|NumGrad|AutoStart|MaxIter)\b", common, re.I):
        raise ValueError("Put stage-specific SCF/gradient/guess options in driver placeholders")
    if not re.search(r"\*\s+xyzfile\s+__CHARGE__\s+\d+\s+orca_coord\.xyz\b", common, re.I):
        raise ValueError("Template must use '* xyzfile __CHARGE__ <multiplicity> orca_coord.xyz'")


def run_job(executable, template, stem, method, charge, guess=None, pc=None,
            frozen=False):
    replacements = {
        "__METHOD__": method + " NoAutoStart",
        "__SCF__": "%scf\n MaxIter " + ("0\n ConvForced false" if frozen else "500\n ConvForced true") + "\nend",
        "__GUESS__": "! MORead\n%moinp " + quoted(guess) if guess else "",
        "__POINTCHARGES__": "%pointcharges " + quoted(pc) if pc else "",
        "__CHARGE__": str(charge),
    }
    contents = template
    for token, value in replacements.items():
        contents = contents.replace(token, value)
    if re.search(r"__[A-Z_]+__", contents):
        raise ValueError("Unresolved placeholder in ORCA template")
    inp = Path(stem + ".inp")
    inp.write_text(contents)
    for suffix in (".engrad", ".gbw"):
        Path(stem + suffix).unlink(missing_ok=True)
    out = Path(stem + ".out")
    with out.open("w") as stream:
        result = subprocess.run([executable, str(inp.resolve())], stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    text = out.read_text()
    if result.returncode or "ORCA TERMINATED NORMALLY" not in text:
        raise RuntimeError(f"ORCA job {stem} failed; inspect {out}")
    if not frozen and not re.search(r"SCF\s+CONVERGED\s+AFTER", text):
        raise RuntimeError(f"ORCA SCF did not converge; inspect {out}")
    if not Path(stem + ".gbw").is_file():
        raise RuntimeError(f"ORCA job {stem} produced no orbitals")
    return text

