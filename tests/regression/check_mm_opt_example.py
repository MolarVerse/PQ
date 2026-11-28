"""Run the ``ar3_mm_opt`` example and a variant of it.

1. The example as shipped converges with the default learning-rate settings.
2. With a small ``nstep`` it does not converge: PQ warns and exits with 0.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT = "run-01.in"


def run_pq(pq, directory, extra):
    text = (directory / INPUT).read_text()
    for key in re.findall(r"^(\S+)\s*=", extra, re.M):
        text = re.sub(rf"^{re.escape(key)}\s*=.*\n", "", text, flags=re.M)
    text += extra
    (directory / "variant.in").write_text(text)
    result = subprocess.run(
        [pq, "variant.in"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def check(name, condition, output, problems):
    if not condition:
        problems.append(f"{name}:\n{output[-1500:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--example", required=True, type=Path)
    args = parser.parse_args()

    problems = []
    cases = [
        ("converges", "", 0, ["Optimizer converged after", "PQ ended normally"], []),
        (
            "warns at nstep",
            "nstep = 2;\n",
            0,
            ["Optimizer did not converge after 2 epochs."],
            ["PQ ended normally"],
        ),
    ]

    for name, extra, expected_code, texts, absent in cases:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "example"
            shutil.copytree(args.example, directory)
            code, output = run_pq(args.pq, directory, extra)

        exit_ok = code != 0 if expected_code is None else code == expected_code
        check(f"{name} (exit code {code})", exit_ok, output, problems)

        for text in texts:
            check(f"{name} ({text!r} missing)", text in output, output, problems)

        for text in absent:
            check(f"{name} ({text!r} printed)", text not in output, output, problems)

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)

    print(f"checked {len(cases)} mm-opt runs, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
