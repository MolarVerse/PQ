#!/usr/bin/env python3
"""PQ DFTB+ driver; arguments: charge read_charges_bin use_pc template pc_file.

Follows the Perl script step by step, including Perl's number
handling, so every written file stays byte-identical.
"""

import math
from pathlib import Path
import re
import subprocess
import sys

ENCODING = "latin-1"
BOHR = 0.529177210903
WS = " \t\n\r\f\v"
NUMBER = re.compile(r"[ \t\n\r\f\v]*([+-]?)(?:(\d+)(\.\d*)?|(\.\d+))([eE][+-]?\d+)?")
SPECIAL = re.compile(r"[ \t\n\r\f\v]*([+-]?)(inf(?:inity)?|nan)", re.I)


class Die(Exception):
    def __init__(self, message, code=255):
        super().__init__(message)
        self.code = code


def num(text):
    """Perl numification: leading number, else 0; integers stay int."""
    if text is None:
        return 0
    special = SPECIAL.match(text)
    if special:
        value = math.nan if special.group(2).lower() == "nan" else math.inf
        return -value if special.group(1) == "-" else value
    match = NUMBER.match(text)
    if not match:
        return 0
    sign, whole, fraction, only_fraction, exponent = match.groups()
    if whole is not None and fraction is None and exponent is None:
        return -int(whole) if sign == "-" else int(whole)
    return float(match.group(0))


def neg(text):
    """Perl unary minus on a string."""
    if text is None:
        return 0
    if re.match(r"[A-Za-z_]", text):
        return num("-" + text)
    if text.startswith("-"):
        return num("+" + text[1:])
    if text.startswith("+"):
        return num("-" + text[1:])
    return -num(text)


def fmt(spec, value):
    """printf with Perl's spelling of non-finite values."""
    if isinstance(value, float) and not math.isfinite(value):
        width = int(re.match(r"%(\d*)", spec).group(1) or 0)
        word = "NaN" if math.isnan(value) else ("-Inf" if value < 0 else "Inf")
        return word.rjust(width)
    return spec % value


def lines(text):
    """Perl readline: split after every newline, keep the newline."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def awk_split(text):
    """Perl `split` / `split ' '`."""
    return re.split(r"[ \t\n\r\f\v]+", text.strip(WS)) if text and text.strip(WS) else []


def regex_split(pattern, text):
    """Perl split /pattern/: keeps a leading empty field, drops trailing ones."""
    if not text:
        return []
    fields = re.split(pattern, text, flags=re.ASCII)
    while fields and fields[-1] == "":
        fields.pop()
    return fields


def field(fields, index):
    return fields[index] if -len(fields) <= index < len(fields) else None


def read(path, progname, label=None):
    try:
        with open(path, encoding=ENCODING, newline="") as stream:
            return stream.read()
    except OSError as error:
        raise Die(f"\n\n\t (-) {progname}: Error opening file {label or path}\n",
                  error.errno or 255) from None


def writer(path, progname, mode="w", label=None):
    try:
        return open(path, mode, encoding=ENCODING, newline="")
    except OSError as error:
        raise Die(f"\n\n\t (-) {progname}: Error opening file {label or path}\n",
                  error.errno or 255) from None


def shell(command):
    return subprocess.run(command, shell=True, check=False).returncode


class FlipFlop:
    """Perl's two-dot range operator on lines."""

    def __init__(self, start, end):
        self.start, self.end, self.on = re.compile(start, re.ASCII), re.compile(end, re.ASCII), False

    def __call__(self, line):
        if not self.on:
            if not self.start.search(line):
                return False
            self.on = True
        if self.end.search(line):
            self.on = False
        return True


def run_dftb(progname):
    status = shell("dftb+ > run.out")
    if status:
        raise Die(f"\n\n\t (-) {progname}: Error executing dftb+.\n", (status >> 8) or 255)
    status = shell("mv detailed.out dftb.out")
    if status:
        raise Die(f"\n\n\t (-) {progname}: Error moving detailed.out to dftb.out.\n",
                  (status >> 8) or 255)


def force_rows(text, flipflop):
    rows = []
    for line in lines(text):
        if flipflop(line) and re.search(r"^\s+-*\d", line, re.ASCII):
            values = awk_split(line)
            n = len(values) - 3
            rows.append(fmt("%27.18e", neg(field(values, n))) + "  "
                        + fmt("%27.18e", neg(field(values, n + 1))) + "  "
                        + fmt("%27.18e", neg(field(values, n + 2))) + "\n")
    return rows


def force_input(content):
    """Turn the SCC input into a single non-SCC force evaluation."""
    flags = re.S | re.ASCII
    content = re.sub(r"Driver\s*=\s*\{\s*\}", lambda m: "Driver = GeometryOptimisation{\n    MaxSteps = 0\n"
                     "    Optimizer = SteepestDescent{}\n}", content, flags=flags)
    if re.search(r"MaxSccIterations\s*=\s*\d+", content, re.ASCII):
        content = re.sub(r"MaxSccIterations\s*=\s*\d+", "MaxSccIterations = 1", content, flags=re.ASCII)
    else:
        content = re.sub(r"(SCC\s*=\s*Yes\s*\n)", lambda m: m.group(1) + "  MaxSccIterations = 1\n",
                         content, flags=re.ASCII)
    if re.search(r"ConvergentSccOnly\s*=\s*\w+", content, re.ASCII):
        content = re.sub(r"ConvergentSccOnly\s*=\s*\w+", "ConvergentSccOnly = No", content, flags=re.ASCII)
    else:
        content = re.sub(r"(MaxSccIterations\s*=\s*\d+\s*\n)",
                         lambda m: m.group(1) + "  ConvergentSccOnly = No\n", content, flags=re.ASCII)
    content = re.sub(r"ReadInitialCharges\s*=\s*\w+", "ReadInitialCharges = Yes", content, flags=re.ASCII)

    blocks = (r"(\s*)ElectricField\s*=\s*\{[^{}]*\}\s*\n?",
              r"(\s*)ElectricField\s*=\s*\{[^{}]*\{[^{}]*\}[^{}]*\}\s*\n?",
              r"(\s*)ElectricField\s*=\s*\{[^{}]*\{[^{}]*\{[^{}]*\}[^{}]*\}[^{}]*\}\s*\n?")
    changed = True
    while changed:
        changed = False
        for pattern in blocks:
            content, count = re.subn(pattern, lambda m: m.group(1), content, flags=flags)
            changed |= count > 0

    rows = regex_split(r"\n", content)
    inside, depth, end = False, 0, -1
    for index, row in enumerate(rows):
        if re.search(r"^\s*Hamiltonian\s*=\s*DFTB\s*\{", row, re.ASCII):
            inside, depth = True, 1
        elif inside:
            depth += row.count("{") - row.count("}")
            if depth == 0:
                end = index
                break
    if end >= 0:
        rows[end:end] = ["  mixer = simple{", "    mixingparameter = 0.0000000001", "  }"]
    content = "\n".join(rows) + "\n"

    if re.search(r"Analysis\s*=\s*\{", content, re.ASCII):
        if not re.search(r"PrintForces\s*=\s*\w+", content, re.ASCII):
            content = re.sub(r"(Analysis\s*=\s*\{[^}]*?)(\s*\})",
                             lambda m: m.group(1) + "\n   PrintForces = Yes" + m.group(2), content, flags=flags)
    else:
        content += "\nAnalysis = {\n    PrintForces = Yes\n}\n"
    return content


def main(argv):
    progname = Path(sys.argv[0]).name
    charge, read_charges, pointcharges, template, pc_file = (list(argv) + [None] * 5)[:5]
    use_pc = num(pointcharges) == 1

    try:
        Path("geom.gen").write_bytes(Path("coords").read_bytes())
    except OSError:
        sys.stderr.write("cat: coords: No such file or directory\n")
        Path("geom.gen").write_bytes(b"")

    geometry = lines(read("geom.gen", progname))
    first = geometry[0].removesuffix("\n") if geometry else None
    second = geometry[1].removesuffix("\n") if len(geometry) > 1 else None
    periodic = field(regex_split(r"\s+", first), 1) == "S" if first is not None else False
    elements = {e for e in regex_split(r"\s+", second) if e != ""} if second is not None else set()

    n_pc = len(lines(read(pc_file, progname, f"{pc_file}'"))) if use_pc else 0

    if num(read_charges) == 0:
        Path("charges.bin").unlink(missing_ok=True)

    template_lines = lines(read(template, progname, f"{template}'"))
    guess = "Yes" if num(read_charges) == 1 else "No"
    hubbard = angular = False
    depth = 0
    out = []
    for line in template_lines:
        if re.search(r"^\s*HubbardDerivs\s*\{", line, re.ASCII):
            hubbard, depth = True, 1
        elif re.search(r"^\s*MaxAngularMomentum\s*\{", line, re.ASCII):
            angular, depth = True, 1
        elif hubbard or angular:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                hubbard = angular = False
        if (hubbard or angular) and depth > 0:
            element = re.search(r"^\s*([A-Z][a-z]?)\s*=", line, re.ASCII)
            if element and element.group(1) not in elements:
                line = re.sub(r"^(\s*)", lambda m: m.group(1) + "# ", line, count=1, flags=re.ASCII)
        line = line.replace("__GUESS__", f"ReadInitialCharges = {guess}", 1)
        line = line.replace("__CHARGE__", charge if charge is not None else "", 1)
        line = line.replace("__N_PC__", str(n_pc), 1)
        out.append(line)
    with writer("dftb_in.hsd", progname) as stream:
        stream.write("".join(out))

    run_dftb(progname)
    detailed = read("dftb.out", progname, "'dftb.out'")
    forces = writer("qm_forces", progname, label="'qm_forces'")
    charges = writer("qm_charges", progname, label="'qm_charges'")
    pc_forces = writer("pc_forces", progname, label="'pc_forces'") if use_pc else None
    stress = writer("stress_tensor", progname, label="'stress_tensor'") if periodic else None

    found = dict(energy=False, forces=False, charges=False, pc_forces=False, stress=False)
    force_ff = FlipFlop(r"^\s*Total Forces", r"^\s*Maximal derivative component:")
    net_ff = FlipFlop(r"^\s*Atom       Net charge", r"^\s*COMPONENT")
    gross_ff = FlipFlop(r"^\s*Atomic gross charges", r"^\s*COMPONENT")
    pc_ff = FlipFlop(r"^\s*Forces on external charges", r"^\s*Volume:")
    stress_ff = FlipFlop(r"^\s*Total stress tensor", r"^\s*Total lattice derivs")
    for line in lines(detailed):
        energy = re.search(r"^\s*Total energy:\s+(.*?)\s+H\s+.*?", line, re.ASCII)
        if energy:
            forces.write(fmt("%20.9f", num(energy.group(1))) + "\n")
            found["energy"] = True
        if not use_pc and force_ff(line) and re.search(r"^\s+-*\d", line, re.ASCII):
            values = awk_split(line)
            n = len(values) - 3
            forces.write(fmt("%27.18e", neg(field(values, n))) + "  "
                         + fmt("%27.18e", neg(field(values, n + 1))) + "  "
                         + fmt("%27.18e", neg(field(values, n + 2))) + "\n")
            found["forces"] = True
        if (net_ff(line) or gross_ff(line)) and re.search(r"^\s+\d+\s+", line, re.ASCII):
            charges.write(fmt("%20.8f", num(field(awk_split(line), 1))) + "\n")
            found["charges"] = True
        if pc_ff(line) and re.search(r"^\s+-*\d", line, re.ASCII):
            values = awk_split(line)
            if pc_forces:
                pc_forces.write("  ".join(fmt("%27.18e", neg(field(values, i))) for i in range(3)) + "\n")
            found["pc_forces"] = True
        if stress_ff(line) and re.search(r"^\s+-*\d", line, re.ASCII):
            values = awk_split(line)
            if stress:
                stress.write("  ".join(fmt("%27.18e", num(field(values, i))) for i in range(3)) + "\n")
            found["stress"] = True
    for stream in (forces, charges, pc_forces, stress):
        if stream:
            stream.close()

    shell("mv dftb.out  dftb.out.prev")
    shell("mv coords coords.prev")

    if use_pc:
        content = read("dftb_in.hsd", progname) if Path("dftb_in.hsd").exists() else ""
        with writer("dftb_in.hsd", progname) as stream:
            stream.write(force_input(content))
        run_dftb(progname)
        detailed = read("dftb.out", progname, "'dftb.out'")
        rows = force_rows(detailed, FlipFlop(r"^\s*Total Forces", r"^\s*Maximal derivative component:"))
        with writer("qm_forces", progname, "a", "'qm_forces'") as stream:
            stream.write("".join(rows))
        found["forces"] |= bool(rows)
        shell("mv dftb.out  dftb.out.prev")

    # DFTB+ includes the QM-point-charge Coulomb energy in its total energy,
    # but PQ adds the same interaction classically
    if use_pc and found["energy"] and found["charges"]:
        geometry = lines(read("geom.gen", progname))
        head = awk_split(geometry[0]) if geometry else []
        n_qm, form = field(head, 0), field(head, 1)
        if form == "F":
            raise Die(f"\n\n\t (-) {progname}: Fractional coordinates are not supported with point charges.\n")
        positions = []
        for index in range(1, int(num(n_qm)) + 1):
            values = awk_split(field(geometry, index + 1) or "")
            positions.append([field(values, i) for i in (2, 3, 4)])
        charge_lines = lines(read("qm_charges", progname, "'qm_charges'"))
        q = [num(field(charge_lines, i)) for i in range(len(positions))]
        e_pc = 0.0
        for line in lines(read(pc_file, progname, f"{pc_file}'")):
            pc = awk_split(line)
            if len(pc) < 4:
                continue
            for i, xyz in enumerate(positions):
                r = math.sqrt((num(xyz[0]) - num(pc[0])) ** 2 + (num(xyz[1]) - num(pc[1])) ** 2
                              + (num(xyz[2]) - num(pc[2])) ** 2) / BOHR
                e_pc += q[i] * num(pc[3]) / r
        force_lines = lines(read("qm_forces", progname, "'qm_forces'"))
        first_line = fmt("%20.9f", num(field(force_lines, 0)) - e_pc) + "\n"
        force_lines = [first_line] + force_lines[1:]
        with writer("qm_forces", progname, label="'qm_forces'") as stream:
            stream.write("".join(force_lines))

    messages = (("energy", "QM energy has not been read!"), ("forces", "QM forces have not been read!"),
                ("charges", "QM charges have not been read!"))
    for key, text in messages:
        if not found[key]:
            sys.stdout.write(f"\n\n\t (-) {progname}: Error encountered while reading dftb.out.\n\t     {text}")
    if use_pc and not found["pc_forces"]:
        sys.stdout.write(f"\n\n\t (-) {progname}: Error encountered while reading dftb.out.\n\t     PC forces have not been read!")
    if periodic and not found["stress"]:
        sys.stdout.write(f"\n\n\t (-) {progname}: Error encountered while reading dftb.out.\n\t     Stress tensor has not been read!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Die as error:
        sys.stdout.flush()
        sys.stderr.write(str(error))
        sys.exit(error.code)
