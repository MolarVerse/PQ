"""Compare the defaults of ``inputFile.rst`` with what PQ really uses.

The user guide states a default for every key (``key = {type} -> default``).
Nothing ties these numbers to the code, so a changed default or a typo in the
guide goes unnoticed. This check runs PQ on the ``h2o_mm`` example with only
the type of the thermostat or manostat set, reads the values it reports in the
setup section of the ``.log`` file, and lists the files it writes when no
``file_prefix`` is given. Every one of these is compared with the number in
the guide, which is parsed from the file, not copied here.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
N_STEPS = 2

# keys whose default is a file name written by an MM-MD run
OUTPUT_FILE_KEYS = [
    "output_file",
    "box_file",
    "charge_file",
    "energy_file",
    "force_file",
    "info_file",
    "instant_energy_file",
    "momentum_file",
    "reference_file",
    "restart_file",
    "stress_file",
    "timings_file",
    "traj_file",
    "vel_file",
    "virial_file",
]

BASE_SETTINGS = (
    "thermostat = berendsen; temp = 300; "
    "manostat = berendsen; pressure = 1.0; "
)

# (name, extra input, [(documented key, section of the log, regex)])
# the regex has one group with the number (or word) the log reports
VALUE_CASES = [
    (
        "berendsen thermostat and manostat",
        BASE_SETTINGS,
        [
            ("t_relaxation", "thermostat", rf"Relaxation time:\s+{NUMBER} ps"),
            ("p_relaxation", "manostat", rf"Relaxation time:\s+{NUMBER} ps"),
            (
                "compressibility",
                "manostat",
                rf"Compressibility:\s+{NUMBER} 1/bar",
            ),
            ("isotropy", "manostat", r"Isotropy:\s+(\w+)"),
            ("rcoulomb", "potential", rf"Coulomb radius cut-off:\s+{NUMBER}"),
            ("nscale", "reset", rf"reset temperature:\s+first\s+{NUMBER} steps"),
            ("nreset", "reset", rf"reset momentum:\s+first\s+{NUMBER} steps"),
            (
                "nreset_angular",
                "reset",
                rf"reset angular momentum:\s+first\s+{NUMBER} steps",
            ),
            (
                "fscale",
                "reset",
                rf"reset temperature:.*every\s+{NUMBER} steps",
            ),
            ("freset", "reset", rf"reset momentum:.*every\s+{NUMBER} steps"),
            (
                "freset_angular",
                "reset",
                rf"reset angular momentum:.*every\s+{NUMBER} steps",
            ),
            (
                "freset_forces",
                "reset",
                rf"reset forces:\s+every\s+{NUMBER} steps",
            ),
        ],
    ),
    (
        "langevin thermostat",
        "thermostat = langevin; temp = 300; ",
        [("friction", "thermostat", rf"Friction:\s+{NUMBER} 1/ps")],
    ),
    (
        "nose-hoover thermostat",
        "thermostat = nh-chain; temp = 300; ",
        [
            (
                "nh_chain_length",
                "thermostat",
                rf"NH chain length:\s+{NUMBER}",
            ),
            (
                "coupling_frequency",
                "thermostat",
                rf"NH coupling freq:\s+{NUMBER} 1/cm",
            ),
        ],
    ),
]

SECTIONS = {
    "thermostat": ("Setup of thermostat", "Setup of Manostat"),
    "manostat": ("Setup of Manostat", "Setup of MM potential"),
    "potential": ("Setup of MM potential", "STARTING SIMULATION"),
    "reset": ("Setup of Reset Kinetics", "Setup of simulation box"),
}


def normalise(key):
    return key.strip().lower().replace("-", "_")


def documented_defaults(guide):
    lines = guide.read_text().splitlines()
    defaults = {}

    for index, line in enumerate(lines):
        if not line.strip().startswith(".. admonition:: Key"):
            continue

        for following in lines[index + 1 : index + 8]:
            match = re.match(
                r"\s+([A-Za-z0-9_\-]+)\*?\s*=\s*\{[^}]*\}[^\n]*?"
                r"(?:\s*->\s*(.*))?$",
                following,
            )
            if match:
                if match.group(2):
                    defaults[normalise(match.group(1))] = match.group(2).strip()
                break

    return defaults


def as_number(text, n_steps):
    """The number of a documented default; ``nstep + 1`` is evaluated."""
    if re.match(r"\s*nstep\s*\+\s*1\b", text):
        return float(n_steps + 1)

    match = re.search(NUMBER, text)
    return float(match.group(1)) if match else None


def run_pq(pq, example, extra_input, with_prefix):
    work = Path(tempfile.mkdtemp(prefix="pq-defaults-"))
    case = work / "case"
    shutil.copytree(example, case)

    text = (case / "run-01.in").read_text()
    text = re.sub(
        r"\b(thermostat|temp|t_relaxation|manostat|pressure|p_relaxation|"
        r"rcoulomb|file_prefix|nstep)\s*=[^;]*;",
        "",
        text,
    )
    prefix = "file_prefix = run; " if with_prefix else ""
    text = text.replace(
        "integrator = v-verlet;",
        f"integrator = v-verlet; nstep = {N_STEPS}; {extra_input}{prefix}",
        1,
    )
    (case / "run-01.in").write_text(text)

    result = subprocess.run(
        [pq, "run-01.in"],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    return case, result


def section(log, name):
    start, end = SECTIONS[name]
    begin = log.find(start)
    finish = log.find(end, begin + 1)
    return log[begin:finish] if begin >= 0 and finish > begin else ""


def check_values(pq, example, defaults, problems):
    checked = 0

    for name, extra, checks in VALUE_CASES:
        case, result = run_pq(pq, example, extra, True)
        log_file = case / "run.log"

        if result.returncode != 0 or not log_file.exists():
            problems.append(f"{name}: PQ failed: {result.stderr.strip()[:200]}")
            shutil.rmtree(case.parent, ignore_errors=True)
            continue

        log = log_file.read_text()

        for key, where, pattern in checks:
            documented = defaults.get(key)
            if documented is None:
                problems.append(f"{name}: no documented default for {key}")
                continue

            match = re.search(pattern, section(log, where))
            if not match:
                problems.append(f"{name}: {key} not reported in the log")
                continue

            reported = match.group(1)
            checked += 1

            if key == "isotropy":
                same = documented.strip('"').lower() == reported.lower()
            else:
                expected = as_number(documented, N_STEPS)
                same = expected is not None and abs(
                    float(reported) - expected
                ) <= 1e-9 * max(1.0, abs(expected))

            if not same:
                problems.append(
                    f"{key}: the guide says {documented}, "
                    f"PQ uses {reported} ({name})"
                )

        shutil.rmtree(case.parent, ignore_errors=True)

    return checked


def check_file_names(pq, example, defaults, problems):
    case, result = run_pq(pq, example, BASE_SETTINGS, False)

    if result.returncode != 0:
        problems.append(f"file names: PQ failed: {result.stderr.strip()[:200]}")
        shutil.rmtree(case.parent, ignore_errors=True)
        return 0

    written = {p.name for p in case.glob("default.*")}
    documented = set()

    for key in OUTPUT_FILE_KEYS:
        value = defaults.get(key)
        if value is None:
            problems.append(f"no documented default for {key}")
            continue
        documented.add(value.strip('"'))

    for name in sorted(documented - written):
        problems.append(f"documented default file {name} is not written")
    for name in sorted(written - documented):
        problems.append(f"PQ writes {name}, which the guide does not document")

    shutil.rmtree(case.parent, ignore_errors=True)
    return len(documented)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument("--guide", required=True, type=Path)
    args = parser.parse_args()

    defaults = documented_defaults(args.guide)
    problems = []

    if len(defaults) < 50:
        problems.append(f"only {len(defaults)} defaults found in {args.guide}")

    checked = check_values(args.pq, args.example, defaults, problems)
    checked += check_file_names(args.pq, args.example, defaults, problems)

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)

    print(f"compared {checked} documented defaults, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
