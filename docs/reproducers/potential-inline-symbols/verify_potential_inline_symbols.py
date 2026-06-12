#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = ROOT / "build-stochastic-rescale-ci"
LIBPOTENTIAL = BUILD_DIR / "src/potential/libpotential.dylib"
TEST_EXE = BUILD_DIR / "tests/src/potential/testBruteForceCellListEquivalence"
BRUTE_SOURCE = ROOT / "src/potential/potentialBruteForce.cpp"
CELL_SOURCE = ROOT / "src/potential/potentialCellList.cpp"


SYMBOLS = (
    "PotentialBruteForce::calculateForces",
    "PotentialCellList::calculateForces",
)


def require(condition, message):
    if not condition:
        raise SystemExit(f"finding not reproduced: {message}")


def command_output(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def demangled_nm(path, extra_flags):
    nm = shutil.which("nm")
    require(nm is not None, "nm is not available")
    raw = command_output(nm, *extra_flags, str(path))
    require(raw.returncode == 0, f"nm failed for {path}: {raw.stderr}")

    cxxfilt = shutil.which("c++filt")
    if cxxfilt is None:
        return raw.stdout

    demangled = subprocess.run(
        [cxxfilt],
        input=raw.stdout,
        text=True,
        capture_output=True,
        check=False,
    )
    require(demangled.returncode == 0, "c++filt failed")
    return demangled.stdout


def main():
    brute_source = BRUTE_SOURCE.read_text()
    cell_source = CELL_SOURCE.read_text()
    require("inline void PotentialBruteForce::" in brute_source, "brute-force override is no longer inline in the .cpp")
    require("inline void PotentialCellList::calculateForces" in cell_source, "cell-list override is no longer inline in the .cpp")

    require(LIBPOTENTIAL.exists(), f"missing build artifact: {LIBPOTENTIAL}")
    require(TEST_EXE.exists(), f"missing test executable: {TEST_EXE}")

    exported = demangled_nm(LIBPOTENTIAL, ("-gU",))
    undefined = demangled_nm(TEST_EXE, ("-u",))

    for symbol in SYMBOLS:
        print(f"checking symbol: {symbol}")
        require(symbol not in exported, f"{symbol} is now exported by libpotential")
        require(symbol in undefined, f"{symbol} is no longer undefined in the test executable")

    run = command_output(
        "ctest",
        "--test-dir",
        str(BUILD_DIR),
        "-R",
        "testBruteForceCellListEquivalence",
        "--output-on-failure",
    )
    print(run.stdout)
    if run.stderr:
        print(run.stderr)
    require(run.returncode != 0, "equivalence test no longer fails in the current build")

    print("finding reproduced: equivalence test binary has unresolved calculateForces overrides and currently fails")


if __name__ == "__main__":
    main()
