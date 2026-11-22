"""Run the constrained ``mm-opt`` examples and check the final geometry.

1. ``h2o_mm_opt_shake``: SHAKE keeps both O-H bonds at 0.9572 Å and the
   angle relaxes to 104.5°.
2. The same water without SHAKE but with stiff bonds reaches the same
   geometry.
"""

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT = "run-01.in"

STIFF_BONDS = """BONDS
1 0.9572 20000.0
END

ANGLES
1 104.5 100.0
END
"""


def run_pq(pq, directory, replace, files):
    text = (directory / INPUT).read_text()
    for old, new in replace.items():
        if old not in text:
            raise ValueError(f"{old!r} not in {INPUT}")
        text = text.replace(old, new)
    (directory / "variant.in").write_text(text)
    for name, content in files.items():
        (directory / name).write_text(content)
    result = subprocess.run(
        [pq, "variant.in"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def read_positions(rst_file):
    positions = []
    for line in rst_file.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0] not in ("Box", "Step"):
            positions.append(tuple(float(value) for value in fields[3:6]))
    return positions


def angle(center, first, second):
    vec1 = [a - b for a, b in zip(first, center)]
    vec2 = [a - b for a, b in zip(second, center)]
    cosine = sum(a * b for a, b in zip(vec1, vec2))
    cosine /= math.dist(first, center) * math.dist(second, center)
    return math.degrees(math.acos(cosine))


def water_geometry(positions):
    oxygen, hydrogen1, hydrogen2 = positions[:3]
    return {
        "O-H1": math.dist(oxygen, hydrogen1),
        "O-H2": math.dist(oxygen, hydrogen2),
        "H-O-H": angle(oxygen, hydrogen1, hydrogen2),
    }


def expect(values, targets, tolerance):
    return [
        f"{key} = {values[key]:.8f}, expected {target} +- {tolerance}"
        for key, target in targets.items()
        if abs(values[key] - target) > tolerance
    ]


def check_shake(positions):
    geometry = water_geometry(positions)
    return expect(geometry, {"O-H1": 0.9572, "O-H2": 0.9572}, 1e-6) + expect(
        geometry, {"H-O-H": 104.5}, 1e-2
    )


def check_stiff(positions):
    geometry = water_geometry(positions)
    return expect(
        geometry, {"O-H1": 0.9572, "O-H2": 0.9572, "H-O-H": 104.5}, 1e-3
    )


CASES = [
    ("shake", "h2o_mm_opt_shake", {}, {}, "h2o-opt.rst", check_shake),
    (
        "stiff bonds without shake",
        "h2o_mm_opt_shake",
        {
            "shake = on;": "shake = off;",
            "nstep = 1000;": "nstep = 20000;",
            "initial-learning-rate = 0.002;": "initial-learning-rate = 0.00004;",
        },
        {"h2o.param": STIFF_BONDS},
        "h2o-opt.rst",
        check_stiff,
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--examples", required=True, type=Path)
    args = parser.parse_args()

    problems = []

    for name, example, replace, files, rst, check in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / example
            shutil.copytree(args.examples / example, directory)
            code, output = run_pq(args.pq, directory, replace, files)

            if code != 0 or "Optimizer converged after" not in output:
                problems.append(f"{name} (exit code {code}):\n{output[-1500:]}")
                continue

            for problem in check(read_positions(directory / rst)):
                problems.append(f"{name}: {problem}")

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)

    print(f"checked {len(CASES)} constrained mm-opt runs, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
