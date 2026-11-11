"""Shared driver for the Python Turbomole interfaces; output matches the Perl scripts byte for byte."""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys

ENCODING = "latin-1"
CONVERGED = re.compile(r"convergence criteria satisfied after\s*\d+\s+iterations", re.ASCII)
FIRST_ITERATION = re.compile(
    r"^\s+1\s+-\d+.\d+\s+-\d+.\d+\s+\d+.\d+\s+\d+.\d+D\+\d+\s+\d+.\d+D", re.ASCII)
CHARGE_ROW = re.compile(r"\s+\d+(\S+)\s+(\S+)", re.ASCII)
PERL_NUMBER = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", re.ASCII)


@dataclass(frozen=True)
class Mode:
    scf: str
    gradient: str
    gradient_done: str
    nopc_out: str
    ricc2: bool = False


HF_DFT = Mode("dscf", "grad", "grad ended normally", "dscf_nopc.out")
RIDFT = Mode("ridft", "rdgrad", "grad ended normally", "ridft_nopc.out")
RICC2 = Mode("dscf", "ricc2", "all done", "dscf_nopc.out", ricc2=True)


class Die(Exception):
    pass


def number(text):
    """Perl numification: leading number or 0."""
    match = PERL_NUMBER.match(text or "")
    return float(match.group(1)) if match else 0.0


def perl_str(value):
    return "%.15g" % value


def read(path):
    with open(path, encoding=ENCODING, newline="") as stream:
        return stream.read()


def lines(path):
    return read(path).splitlines(keepends=True)


def write(path, text, append=False):
    with open(path, "a" if append else "w", encoding=ENCODING, newline="") as stream:
        stream.write(text)


def tee(command, out_file, append=False, stdin=None):
    """Like `command 2>&1 | tee [-a] out_file`, returning the captured output."""
    try:
        done = subprocess.run([command], stdin=stdin, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        text = done.stdout.decode(ENCODING)
    except OSError as error:
        text = f"sh: {command}: {error.strerror}\n"
    write(out_file, text, append)
    return text


def copy_if_exists(source, target):
    if Path(source).is_file():
        shutil.copyfile(source, target)


def open_or_die(path, progname, label=None):
    try:
        return lines(path)
    except OSError:
        raise Die(f"\n\n\t (-) {progname}: Error opening file '{label or path}'\n") from None


def perl_split(text):
    """Perl split(/\\s+/, $line): keeps a leading empty field, drops trailing ones."""
    fields = re.split(r"\s+", text, flags=re.ASCII)
    while fields and fields[-1] == "":
        fields.pop()
    return fields


def insert_point_charges(point_charge_file, progname):
    kept = []
    for line in open_or_die("control", progname):
        if re.match(r"^\s*\$end", line, re.ASCII):
            break
        kept.append(line)
    rows = open_or_die(point_charge_file, progname)
    block = []
    for line in rows:
        fields = line.split() + [""] * 4
        block.append("%14.8f %14.8f %14.8f %10.6f\n" % tuple(number(f) for f in fields[:4]))
    write("control", "".join(kept) + "$point_charges\n" + "".join(block) + "$end\n")


def remove_point_charges(progname):
    contents, flag = [], False
    for line in open_or_die("control", progname):
        if line.startswith("$point_charges"):
            flag = True
        if flag and line.startswith("$") and not line.startswith("$point_charges"):
            flag = False
        if not flag:
            contents.append(line)
    write("control", "".join(contents))


def total_energy(path, progname):
    energy = None
    for line in open_or_die(path, progname):
        if re.match(r"^\s+:\s+kinetic", line, re.ASCII):
            break
        line = line.replace("D", "E")
        if re.match(r"^\s+\|\s+total", line, re.ASCII):
            fields = line.split()
            energy = fields[4] if len(fields) > 4 else None
    return energy


def first_iteration_energy(path, progname):
    energy = None
    for line in open_or_die(path, progname):
        if FIRST_ITERATION.match(line):
            energy = line.split()[1]
    return energy


def gradient_rows(progname):
    rows = []
    for line in open_or_die("gradient", progname):
        if re.match(r"^\s*\$grad", line, re.ASCII) or re.match(r"^\s*cycle =", line, re.ASCII):
            continue
        if re.match(r"^\s*\$end", line, re.ASCII):
            break
        fields = line.replace("D", "E").split()
        if len(fields) == 3:
            rows.append("%20.14G  %20.14G  %20.14G\n" % tuple(number(f) for f in fields))
    return rows


def mulliken(path, progname, ricc2):
    charges, flag, found = [], False, not ricc2
    for line in open_or_die(path, progname):
        if not found:
            if re.search(r"\s*R\s+I\s+C\s+C\s+2", line, re.ASCII):
                found = True
            continue
        if "from total density versus population analysis" in line:
            break
        if "atom      charge" in line:
            flag = True
        if flag and CHARGE_ROW.search(line):
            item = perl_split(CHARGE_ROW.sub(r"\1 \2", line, count=1))
            charges.append("%12.5f\n" % number(item[1] if len(item) > 1 else ""))
    return charges


def run(mode, argv):
    progname = Path(sys.argv[0]).name
    charge, run_define, use_pc, template, point_charge_file = (argv + [""] * 5)[:5]
    use_pc = number(use_pc) == 1

    if number(run_define) == 1:
        for name in ("control", "basis", "mos", "alpha", "beta", "tmp.input") + \
                (("CCl0o-1--1---0",) if mode.ricc2 else ()):
            Path(name).unlink(missing_ok=True)
        label = template if mode.ricc2 else "tm_define.template"
        template_lines = open_or_die(template, progname, label)
        write("tm_define.in", "".join(line.replace("__CHARGE__", charge, 1)
                                      for line in template_lines))
        with open("tm_define.in", "rb") as stdin:
            tee("define", "tm.out", stdin=stdin)

    if use_pc:
        insert_point_charges(point_charge_file, progname)

    if not CONVERGED.search(tee(mode.scf, "tm.out")):
        return 1

    if use_pc:
        remove_point_charges(progname)
        for name in ("control", "mos", "alpha", "beta"):
            copy_if_exists(name, name + ".bak")
        write("control", "".join(
            "$scfiterlimit      1\n" if re.match(r"^\s*\$scfiterlimit", line, re.ASCII) else line
            for line in open_or_die("control", progname)))
        tee(mode.scf, mode.nopc_out)

    if mode.ricc2:
        energy = total_energy(mode.nopc_out if use_pc else "tm.out", progname)
    elif use_pc:
        energy = first_iteration_energy(mode.nopc_out, progname)
    else:
        energy = total_energy("tm.out", progname)

    if use_pc:
        for name in ("control", "mos", "alpha", "beta"):
            copy_if_exists(name + ".bak", name)

    if energy is None or not re.search(r"\d", energy, re.ASCII):
        raise Die(f"\n\n\t (-) {progname}: no SCF energy found in Turbomole output\n")

    Path("gradient").unlink(missing_ok=True)
    if mode.gradient_done not in tee(mode.gradient, "tm.out", append=True):
        return 1

    if mode.ricc2:
        last = None
        for line in open_or_die("energy", progname):
            if "$" not in line:
                last = line
        fields = (last or "").split()
        energy = perl_str(number(energy) + number(fields[4] if len(fields) > 4 else ""))

    write("qm_forces", energy + "\n")
    write("qm_forces", "".join(gradient_rows(progname)), append=True)

    charge_file = "tm.out" if mode.ricc2 or not use_pc else mode.nopc_out
    rows = mulliken(charge_file, progname, mode.ricc2)
    write("qm_charges", "".join(rows))
    return 0


def main(mode):
    try:
        sys.exit(run(mode, sys.argv[1:]))
    except Die as error:
        sys.stderr.write(str(error))
        sys.exit(255)
