"""Run the Perl and Python Turbomole drivers against a fake Turbomole; usage: <scripts dir>."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(sys.argv.pop(1)).resolve()

COMMON = r'''#!/usr/bin/env python3
import os
from pathlib import Path


def mulliken(o, h):
    return ("  atom      charge    n(s)\n"
            f"    1o     {o:.5f}   3.8\n"
            f"    2h     {h:.5f}   0.7\n"
            f"    3h     {h:.5f}   0.7\n"
            " from total density versus population analysis\n")


def gradient():
    Path("gradient").write_text(
        "$grad          cartesian gradients\n"
        "  cycle =      1    SCF energy =   -76.01   |dE/dxyz| =  0.01\n"
        "    0.0000000    0.0000000    0.0000000      o\n"
        "    1.4300000    0.0000000    0.9500000      h\n"
        "   -1.4300000    0.0000000    0.9500000      h\n"
        "   0.1000000D-01   0.0000000D+00   0.2000000D-01\n"
        "  -0.5000000D-02   0.0000000D+00  -0.1000000D-01\n"
        "  -0.5000000D-02   0.0000000D+00  -0.1000000D-01\n"
        "$end\n")
'''
FAKE = {
    "define": COMMON + r'''
import sys
sys.stdin.read()
Path("control").write_text("$title\n$scfiterlimit     30\n$end\n")
Path("mos").write_text("$scfmo\n$end\n")
''',
    "dscf": COMMON + r'''
control = Path("control").read_text()
if "$scfiterlimit      1" in control:
    if os.environ.get("FAKE_TM_BREAK") != "nopc":
        print("   1  -76.0050000000  -120.0000  44.1000  0.000D+00  0.100D-01")
        print("  |  total energy      =    -.760050000000D+02  |")
    print("  :  kinetic energy    =     75.9000000000  :")
    print(mulliken(-0.64, 0.32))
else:
    o, h = (-0.70, 0.35) if "$point_charges" in control else (-0.66, 0.33)
    if os.environ.get("FAKE_TM_BREAK") != "scf":
        print(" convergence criteria satisfied after  7 iterations")
    print("  |  total energy      =    -76.0100000000  |")
    print("  :  kinetic energy    =     75.9000000000  :")
    print(mulliken(o, h))
''',
    "grad": COMMON + '''
gradient()
if os.environ.get("FAKE_TM_BREAK") != "grad":
    print("grad ended normally")
''',
    "ricc2": COMMON + r'''
gradient()
Path("energy").write_text("$energy\n      1   -76.01   0.0   0.0   -0.2\n$end\n")
print("       R I C C 2")
print(mulliken(-0.60, 0.30))
if os.environ.get("FAKE_TM_BREAK") != "grad":
    print("  ricc2 : all done")
''',
}
FAKE["ridft"] = FAKE["dscf"]
FAKE["rdgrad"] = FAKE["grad"]

PAIRS = (("turbomole_hf-dft", "turbomole_hf_dft.py"),
         ("turbomole_ridft", "turbomole_ridft.py"),
         ("turbomole_ricc2", "turbomole_ricc2.py"))
SCF_SCRIPTS = [s for pair in PAIRS[:2] for s in pair]
ALL_SCRIPTS = [s for pair in PAIRS for s in pair]

GRADIENT = [[0.01, 0.0, 0.02], [-0.005, 0.0, -0.01], [-0.005, 0.0, -0.01]]


class TurbomoleScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pq-tm-")
        self.work = Path(self.temp.name)
        bin_dir = self.work / "bin"
        bin_dir.mkdir()
        for name, body in FAKE.items():
            (bin_dir / name).write_text(body)
            (bin_dir / name).chmod(0o755)
        (self.work / "tm.template").write_text("charge __CHARGE__\n")
        (self.work / "pc").write_text("5.0 0.0 0.0 -0.8\n")
        self.env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, script, use_pc, define=True, work=None, **env):
        interpreter = sys.executable if script.endswith(".py") else "perl"
        return subprocess.run(
            [interpreter, str(SCRIPTS / script), "0", "1" if define else "0",
             "1" if use_pc else "0", "tm.template", "pc"],
            cwd=work or self.work, env={**self.env, **env}, capture_output=True, text=True,
            check=False)

    def check(self, script, use_pc, energy, charges):
        (self.work / "qm_forces").unlink(missing_ok=True)
        done = self.run_script(script, use_pc)
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = (self.work / "qm_forces").read_text().splitlines()
        self.assertAlmostEqual(float(lines[0]), energy, places=10)
        gradient = [[float(v) for v in line.split()] for line in lines[1:]]
        self.assertEqual(len(gradient), 3)
        for row, want in zip(gradient, GRADIENT):
            for value, expected in zip(row, want):
                self.assertAlmostEqual(value, expected, places=12)
        q = [float(v) for v in (self.work / "qm_charges").read_text().split()]
        self.assertEqual(q, charges)
        if use_pc:
            control = (self.work / "control").read_text()
            self.assertNotIn("$point_charges", control)
            self.assertIn("$scfiterlimit     30", control)

    def test_scf_pure(self):
        for script in SCF_SCRIPTS:
            self.check(script, False, -76.01, [-0.66, 0.33, 0.33])

    def test_scf_point_charges_use_field_free_energy_and_charges(self):
        for script in SCF_SCRIPTS:
            self.check(script, True, -76.005, [-0.64, 0.32, 0.32])

    def test_ricc2(self):
        for script in ("turbomole_ricc2", "turbomole_ricc2.py"):
            self.check(script, False, -76.21, [-0.60, 0.30, 0.30])
            self.check(script, True, -76.205, [-0.60, 0.30, 0.30])

    def test_missing_field_free_energy_fails(self):
        for script in ALL_SCRIPTS:
            (self.work / "qm_forces").unlink(missing_ok=True)
            done = self.run_script(script, True, FAKE_TM_BREAK="nopc")
            self.assertNotEqual(done.returncode, 0, script)
            self.assertIn("no SCF energy found", done.stderr)
            self.assertFalse((self.work / "qm_forces").exists(), script)

    def _snapshot(self, script, steps, **env):
        work = self.work / script.replace(".", "_")
        shutil.copytree(self.work, work, ignore=shutil.ignore_patterns("bin", "turbomole_*"))
        codes = [self.run_script(script, pc, define, work=work, **env).returncode
                 for pc, define in steps]
        files = {p.name: p.read_bytes() for p in sorted(work.iterdir()) if p.is_file()}
        return codes, files

    def test_python_matches_perl_byte_for_byte(self):
        sequences = {
            "pure": [(False, True), (False, False)],
            "point charges": [(True, True), (True, False), (False, False)],
        }
        for perl, python in PAIRS:
            for name, steps in sequences.items():
                for broken in ("", "scf", "nopc", "grad"):
                    with self.subTest(script=python, case=name, broken=broken):
                        old = self._snapshot(perl, steps, FAKE_TM_BREAK=broken)
                        new = self._snapshot(python, steps, FAKE_TM_BREAK=broken)
                        self.assertEqual([c != 0 for c in old[0]], [c != 0 for c in new[0]])
                        self.assertEqual(old[1].keys(), new[1].keys())
                        for key in old[1]:
                            self.assertEqual(old[1][key], new[1][key], key)
                        shutil.rmtree(self.work / perl.replace(".", "_"))
                        shutil.rmtree(self.work / python.replace(".", "_"))


if __name__ == "__main__":
    unittest.main()
