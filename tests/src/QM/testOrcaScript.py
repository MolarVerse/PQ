"""Exercise the ORCA subprocess/file contract without a licensed ORCA install."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(sys.argv.pop(1)).resolve()
TEMPLATE = Path(sys.argv.pop(1)).read_text()
FAKE = r'''#!/usr/bin/env python3
from pathlib import Path
import os
import sys
inp = Path(sys.argv[1])
text = inp.read_text()
stem = inp.stem
with Path("calls").open("a") as f:
    f.write(stem + "\n")
if os.environ.get("FAIL_STAGE") == stem:
    sys.exit(1)
assert "__" not in text
assert "%pointcharges" not in text or stem == "orca_polarized"
if stem != "orca_polarized" and "NoIter" in text:
    assert '%moinp "' in text and 'orca_polarized.gbw"' in text
Path(stem + ".gbw").write_text(stem)
print("ORCA TERMINATED NORMALLY")
print("SCF CONVERGED AFTER 5 CYCLES")
if stem == "orca_frozen_hf":
    if os.environ.get("BAD_RESULT") != "energy":
        print("FINAL SINGLE POINT ENERGY -75.0")
else:
    print("FINAL SINGLE POINT ENERGY -99.0")
if stem == "orca_mp2":
    print("RI-MP2 CORRELATION ENERGY: -0.2 Eh")
    if "NoIter" in text:
        e = -0.2
    else:
        e = -99.0
    if os.environ.get("BAD_RESULT") != "gradient":
        Path(stem + ".engrad").write_text(
            f"# test\n2\n{e}\n0.1\n0.2\n0.3\n-0.1\n-0.2\n-0.3\n1 0 0 0\n1 0 0 1\n")
    print("MULLIKEN ATOMIC CHARGES\n0 H: 0.9\n1 H: -0.9")
    print("UNRELAXED MP2 DENSITY POPULATION ANALYSIS")
    print("MULLIKEN ATOMIC CHARGES\n0 H: 0.8\n1 H: -0.8")
    print("RELAXED MP2 DENSITY POPULATION ANALYSIS")
    if os.environ.get("BAD_RESULT") != "charges":
        print("MULLIKEN ATOMIC CHARGES\n0 H: 0.1\n1 H: -0.1")
'''


class OrcaScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pq orca; ")
        self.work = Path(self.temp.name)
        (self.work / "orca.template").write_text(TEMPLATE)
        (self.work / "orca_coord.xyz").write_text("2\n\nH 0 0 0\nH 0 0 1\n")
        (self.work / "point charges").write_text("1\n0.5 0 0 5\n")
        self.exe = self.work / "fake orca"
        self.exe.write_text(FAKE)
        self.exe.chmod(0o755)
        self.env = dict(os.environ, ORCA_EXE=str(self.exe))

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, pc=True, reset=True, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPT), "0", str(int(reset)),
                                 str(int(pc)), "orca.template", "point charges"],
                                cwd=self.work, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def test_three_stage_energy_and_relaxed_charges(self):
        self.run_script()
        self.assertEqual((self.work / "calls").read_text().splitlines(),
                         ["orca_polarized", "orca_frozen_hf", "orca_mp2"])
        self.assertAlmostEqual(float((self.work / "qm_forces").read_text().split()[0]), -75.2)
        self.assertEqual([float(x) for x in (self.work / "qm_charges").read_text().split()], [0.1, -0.1])
        self.assertNotIn("CalcGuessEnergy", (self.work / "orca_mp2.inp").read_text())
        self.assertIn("CalcGuessEnergy", (self.work / "orca_frozen_hf.inp").read_text())
        self.assertEqual((self.work / "orca_previous.gbw").read_text(), "orca_polarized")

    def test_pure_qm_uses_one_converged_gradient_job(self):
        self.run_script(pc=False)
        self.assertEqual((self.work / "calls").read_text(), "orca_mp2\n")
        self.assertAlmostEqual(float((self.work / "qm_forces").read_text().split()[0]), -99)
        self.assertNotIn("NoIter", (self.work / "orca_mp2.inp").read_text())

    def test_restart_reuses_only_successful_compatible_reference(self):
        self.run_script()
        self.run_script(reset=False)
        self.assertIn('orca_previous.gbw"', (self.work / "orca_polarized.inp").read_text())
        self.run_script(reset=True)
        self.assertNotIn("MORead", (self.work / "orca_polarized.inp").read_text())
        template = self.work / "orca.template"
        template.write_text(TEMPLATE + "\n# changed basis settings\n")
        self.run_script(reset=False)
        self.assertNotIn("MORead", (self.work / "orca_polarized.inp").read_text())

    def test_failure_in_each_stage_removes_results_and_invalidates_restart(self):
        for stage in ("orca_polarized", "orca_frozen_hf", "orca_mp2"):
            with self.subTest(stage=stage):
                self.run_script()
                self.env["FAIL_STAGE"] = stage
                self.run_script(reset=False, ok=False)
                self.assertFalse((self.work / "qm_forces").exists())
                self.assertFalse((self.work / "qm_charges").exists())
                self.assertFalse((self.work / "orca_previous.json").exists())
                del self.env["FAIL_STAGE"]

    def test_missing_results_are_rejected_even_after_successful_previous_step(self):
        for result in ("energy", "gradient", "charges"):
            with self.subTest(result=result):
                self.run_script()
                self.env["BAD_RESULT"] = result
                self.run_script(reset=False, ok=False)
                self.assertFalse((self.work / "qm_forces").exists())
                self.assertFalse((self.work / "qm_charges").exists())
                del self.env["BAD_RESULT"]

    def test_pointcharge_count_mismatch(self):
        (self.work / "point charges").write_text("2\n0.5 0 0 5\n")
        self.run_script(ok=False)
        self.assertFalse((self.work / "calls").exists())

    def test_template_cannot_override_stage_control(self):
        for extra in ("%scf MaxIter 100 end", "%pointcharges \"other\"", "$new_job", "! NoIter"):
            with self.subTest(extra=extra):
                (self.work / "orca.template").write_text(TEMPLATE + '\n' + extra)
                self.run_script(ok=False)
                self.assertFalse((self.work / "calls").exists())

    def test_missing_executable(self):
        self.env["ORCA_EXE"] = str(self.work / "missing")
        self.run_script(ok=False)
        self.assertFalse((self.work / "qm_forces").exists())


if __name__ == "__main__":
    unittest.main()
