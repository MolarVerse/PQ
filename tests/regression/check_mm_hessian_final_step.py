"""Check that the mm-hessian pre-optimization writes its converged step.

Runs ``oh_mm_hessian/run-01.in`` and reads N from the "Optimizer converged
after N epochs." line. The last ``.rst`` must be step N, the geometry the
Hessian is built at.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT = "run-01.in"
RST_FILE = "oh-opt.rst"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--example", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "example"
        shutil.copytree(args.example, directory)
        result = subprocess.run(
            [args.pq, INPUT],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"FAIL PQ exited with {result.returncode}:\n{output[-1500:]}")
            return 1

        converged = re.search(r"Optimizer converged after (\d+) epochs\.", output)
        if not converged:
            print(f"FAIL no convergence message:\n{output[-1500:]}")
            return 1

        rst = (directory / RST_FILE).read_text()
        step = re.search(r"^Step (\d+)", rst, re.M)

    epochs = int(converged.group(1))
    if step is None or int(step.group(1)) != epochs:
        found = step.group(1) if step else "none"
        print(f"FAIL last rst is step {found}, converged step is {epochs}")
        return 1

    print(f"rst is the converged step {epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
