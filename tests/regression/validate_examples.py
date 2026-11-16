"""Validate every input file of the examples directory with ``PQ --validate``.

Each input is checked in the portable scope (input semantics, bundled script
names, values, keywords; independent of the build) and in the installed scope
(files the input refers to exist, QM methods are compiled in). Two outcomes of
the installed scope are not errors: an input that needs ASE in a build without
ASE, and a start file that another input of the same example writes (the
second step of a chained run).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

NOT_COMPILED = re.compile(r"requires ASE support")
START_FILE = re.compile(r'Invalid value "([^"]+)" for key "start_file"')


def file_prefix(text):
    match = re.search(r"^\s*file_prefix\s*=\s*([^;#\s]+)", text, re.M)
    return match.group(1) if match else None


def run_validate(pq, directory, name, scope):
    result = subprocess.run(
        [pq, "--validate", name, f"--scope={scope}"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def check_example(pq, directory, inputs):
    problems = []
    written = {file_prefix(p.read_text()) for p in inputs}
    written = {f"{prefix}.rst" for prefix in written if prefix}

    for path in inputs:
        code, message = run_validate(pq, directory, path.name, "portable")
        if code != 0:
            problems.append(f"{directory.name}/{path.name} (portable): {message}")
            continue

        code, message = run_validate(pq, directory, path.name, "installed")
        if code == 0 or NOT_COMPILED.search(message):
            continue

        match = START_FILE.search(message)
        if match and match.group(1) in written:
            continue

        problems.append(f"{directory.name}/{path.name} (installed): {message}")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--examples", required=True, type=Path)
    args = parser.parse_args()

    problems = []
    checked = 0

    for directory in sorted(p for p in args.examples.iterdir() if p.is_dir()):
        inputs = sorted(
            p
            for p in directory.glob("*.in")
            if re.search(r"^\s*jobtype\s*=", p.read_text(), re.M)
        )
        checked += len(inputs)
        problems += check_example(args.pq, directory, inputs)

    if checked == 0:
        problems.append(f"no example input found in {args.examples}")

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)

    print(f"checked {checked} example inputs, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
