"""Run the constrained ``mm-opt`` examples and check the final geometry.

1. ``h2o_mm_opt_shake``: SHAKE keeps both O-H bonds at 0.9572 Å and the
   angle relaxes to 104.5°.
2. The same water without SHAKE but with stiff bonds reaches the same
   geometry.
3. ``h2o_mm_opt_mshake``: both waters of the dimer stay rigid with M-SHAKE.
4. The dimer with SHAKE on all three distances of each water reaches the same
   energy.
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

RIGID_SHAKE = """SHAKE
1 2 0.957199
1 3 0.957199
2 3 1.5139
4 5 0.957199
4 6 0.957199
5 6 1.5139
END
"""

# from mshake.top
RIGID_OH = math.hypot(0.585882, 0.756950)
RIGID_HH = 2 * 0.756950
DIMER_ENERGY = -6.46783


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


def read_energy(en_file):
    return float(en_file.read_text().splitlines()[-1].split()[3])


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


def check_shake(positions, _energy):
    geometry = water_geometry(positions)
    return expect(geometry, {"O-H1": 0.9572, "O-H2": 0.9572}, 1e-6) + expect(
        geometry, {"H-O-H": 104.5}, 1e-2
    )


def check_stiff(positions, _energy):
    geometry = water_geometry(positions)
    return expect(
        geometry, {"O-H1": 0.9572, "O-H2": 0.9572, "H-O-H": 104.5}, 1e-3
    )


def check_rigid_dimer(positions, energy):
    problems = []
    for first in (0, 3):
        oxygen, hydrogen1, hydrogen2 = positions[first : first + 3]
        distances = {
            "O-H1": math.dist(oxygen, hydrogen1),
            "O-H2": math.dist(oxygen, hydrogen2),
            "H-H": math.dist(hydrogen1, hydrogen2),
        }
        targets = {"O-H1": RIGID_OH, "O-H2": RIGID_OH, "H-H": RIGID_HH}
        problems += expect(distances, targets, 1e-6)
    return problems + expect({"E": energy}, {"E": DIMER_ENERGY}, 1e-4)


CASES = [
    ("shake", "h2o_mm_opt_shake", {}, {}, "h2o-opt", check_shake),
    (
        "stiff bonds without shake",
        "h2o_mm_opt_shake",
        {
            "shake = on;": "shake = off;",
            "nstep = 1000;": "nstep = 20000;",
            "initial-learning-rate = 0.002;": "initial-learning-rate = 0.00004;",
        },
        {"h2o.param": STIFF_BONDS},
        "h2o-opt",
        check_stiff,
    ),
    ("mshake", "h2o_mm_opt_mshake", {}, {}, "h2o-dimer-opt", check_rigid_dimer),
    (
        "shake on all water distances",
        "h2o_mm_opt_mshake",
        {
            "shake = mshake;": "shake = on;\nshake-iter = 100;\nrattle-iter = 100;",
            "mshake_file = mshake.top;\n": "",
        },
        {"shake.top": RIGID_SHAKE},
        "h2o-dimer-opt",
        check_rigid_dimer,
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--examples", required=True, type=Path)
    args = parser.parse_args()

    problems = []

    for name, example, replace, files, prefix, check in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / example
            shutil.copytree(args.examples / example, directory)
            code, output = run_pq(args.pq, directory, replace, files)

            if code != 0 or "Optimizer converged after" not in output:
                problems.append(f"{name} (exit code {code}):\n{output[-1500:]}")
                continue

            positions = read_positions(directory / f"{prefix}.rst")
            energy = read_energy(directory / f"{prefix}.en")
            for problem in check(positions, energy):
                problems.append(f"{name}: {problem}")

    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)

    print(f"checked {len(CASES)} constrained mm-opt runs, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
