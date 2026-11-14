"""HF/DFT subprocess contract tests using a deterministic fake ORCA."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(sys.argv.pop(1)).resolve()
EXAMPLES = Path(sys.argv.pop(1)).resolve()
FAKE = r'''#!/usr/bin/env python3
import os
import re
from pathlib import Path
import sys
inp = Path(sys.argv[1])
text, stem = inp.read_text(), inp.stem
with Path('calls').open('a') as stream:
    stream.write(stem + '\n')
assert not re.search(r'__[A-Z]+__', text)
assert '%pointcharges' not in text or stem.endswith('_polarized')
if os.environ.get('FAIL_STAGE') == stem:
    sys.exit(1)
if 'NoIter' in text:
    assert stem.endswith('_frozen')
    assert 'CalcGuessEnergy' in text and 'EnGrad' in text
    assert f'orca_{stem.split("_")[1]}_polarized.gbw"' in text
    assert 'MaxIter 0' in text
mode = 'hf' if 'HF' in text.splitlines()[1].split() else 'dft'
e = -1.2 if mode == 'hf' else -1.4
if stem.endswith('_polarized'):
    e = -99.0  # Deliberately distinguish the embedded energy from the returned one.
if stem.endswith('_frozen'):
    e += 0.02
bad = os.environ.get('BAD_RESULT', '')
if bad != 'gbw':
    Path(stem + '.gbw').write_text(stem)
if bad != 'termination':
    print('ORCA TERMINATED NORMALLY')
if bad != 'convergence':
    print('SCF CONVERGED AFTER 4 CYCLES')
if bad != 'energy':
    print(f'FINAL SINGLE POINT ENERGY {e}')
if bad == 'wrong_method':
    print('Density Functional Method .... DFT(GTOs)' if mode == 'hf'
          else 'Ab initio Hamiltonian Method .... Hartree-Fock(GTOs)')
if bad == 'correlation':
    print('RI-MP2 CORRELATION ENERGY: -0.1 Eh')
if 'EnGrad' in text and bad != 'gradient':
    ge = e + 1 if bad == 'mismatch' else e
    n = 1 if bad == 'count' else 2
    g = 'nan' if bad == 'nan' else '0.1'
    Path(stem + '.engrad').write_text(
        f'# native format\n{n}\n{ge}\n{g}\n0.2\n0.3\n-0.1\n-0.2\n-0.3\n1 0 0 0\n1 0 0 1\n')
if bad != 'charges':
    print('MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS')
    print('-----------------------')
    print('0 H: 0.1 0.5')
    if bad != 'partial_charges':
        print('1 H: -0.1 -0.5' if bad != 'charge_sum' else '1 H: 0.1 -0.5')
'''


class OrcaScfScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pq scf; ')
        self.work = Path(self.temp.name)
        (self.work / 'orca_coord.xyz').write_text('2\n\nH 0 0 0\nH 0 0 1\n')
        (self.work / 'point charges').write_text('1\n0.5 0 0 5\n')
        self.exe = self.work / 'fake orca'
        self.exe.write_text(FAKE)
        self.exe.chmod(0o755)
        self.env = dict(os.environ, ORCA_EXE=str(self.exe), PYTHONDONTWRITEBYTECODE='1')
        self.mode = 'hf'
        self.use_template('hf')

    def tearDown(self):
        self.temp.cleanup()

    def use_template(self, mode):
        self.mode = mode
        (self.work / 'orca.template').write_text(
            (EXAMPLES / f'orca_{mode}.template').read_text())

    def run_script(self, pc=True, reset=True, ok=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / f'orca_{self.mode}.py'), '0',
             str(int(reset)), str(int(pc)), 'orca.template', 'point charges'],
            cwd=self.work, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def test_pure_hf_and_dft_use_one_converged_gradient(self):
        for mode, expected in [('hf', -1.2), ('dft', -1.4)]:
            with self.subTest(mode=mode):
                self.use_template(mode)
                (self.work / 'calls').unlink(missing_ok=True)
                self.run_script(pc=False)
                self.assertEqual((self.work / 'calls').read_text(), f'orca_{mode}\n')
                self.assertAlmostEqual(float((self.work / 'qm_forces').read_text().split()[0]), expected)
                self.assertNotIn('NoIter', (self.work / f'orca_{mode}.inp').read_text())

    def test_embedding_uses_two_stages_and_no_field_energy(self):
        for mode, expected in [('hf', -1.18), ('dft', -1.38)]:
            with self.subTest(mode=mode):
                self.use_template(mode)
                (self.work / 'calls').unlink(missing_ok=True)
                self.run_script()
                self.assertEqual((self.work / 'calls').read_text().splitlines(),
                                 [f'orca_{mode}_polarized', f'orca_{mode}_frozen'])
                self.assertAlmostEqual(float((self.work / 'qm_forces').read_text().split()[0]), expected)
                self.assertEqual([float(x) for x in (self.work / 'qm_charges').read_text().split()], [0.1, -0.1])
                self.assertEqual((self.work / 'orca_previous.gbw').read_text(), f'orca_{mode}_polarized')
                self.assertNotIn('%pointcharges', (self.work / f'orca_{mode}_frozen.inp').read_text())

    def test_restart_reset_template_and_embedding_mode_changes(self):
        for mode in ('hf', 'dft'):
            self.use_template(mode)
            self.run_script()
            self.run_script(reset=False)
            inp = self.work / f'orca_{mode}_polarized.inp'
            self.assertIn('orca_previous.gbw"', inp.read_text())
            self.run_script(reset=True)
            self.assertNotIn('MORead', inp.read_text())
            template = self.work / 'orca.template'
            template.write_text(template.read_text() + '\n# change\n')
            self.run_script(reset=False)
            self.assertNotIn('MORead', inp.read_text())
            self.run_script(pc=False, reset=False)
            self.assertNotIn('MORead', (self.work / f'orca_{mode}.inp').read_text())

    def test_failure_in_each_stage_invalidates_results_and_restart(self):
        for mode in ('hf', 'dft'):
            self.use_template(mode)
            for stage in (f'orca_{mode}_polarized', f'orca_{mode}_frozen'):
                self.run_script()
                self.env['FAIL_STAGE'] = stage
                self.run_script(reset=False, ok=False)
                del self.env['FAIL_STAGE']
                for name in ('qm_forces', 'qm_charges', 'orca_previous.json'):
                    self.assertFalse((self.work / name).exists())

    def test_incomplete_nonfinite_and_inconsistent_results_fail(self):
        for mode in ('hf', 'dft'):
            self.use_template(mode)
            for bad in ('termination', 'convergence', 'gbw', 'energy', 'gradient',
                        'mismatch', 'nan', 'count', 'charges', 'partial_charges',
                        'charge_sum', 'correlation', 'wrong_method'):
                with self.subTest(mode=mode, bad=bad):
                    self.run_script(pc=False)
                    self.env['BAD_RESULT'] = bad
                    self.run_script(pc=False, reset=False, ok=False)
                    del self.env['BAD_RESULT']
                    self.assertFalse((self.work / 'qm_forces').exists())
                    self.assertFalse((self.work / 'qm_charges').exists())

    def test_wrong_reference_and_post_scf_templates_fail_before_execution(self):
        for mode, reference in [('hf', 'RKS'), ('dft', 'RHF')]:
            self.use_template(mode)
            template = self.work / 'orca.template'
            template.write_text(template.read_text() + f'\n! {reference}\n')
            self.run_script(ok=False)
        for mode in ('hf', 'dft'):
            for extra in ('! RI-MP2', '%mp2 Density relaxed end', '! B2PLYP', '%scf MaxIter 1 end'):
                self.use_template(mode)
                template = self.work / 'orca.template'
                template.write_text(template.read_text() + '\n' + extra)
                self.run_script(ok=False)
        self.assertFalse((self.work / 'calls').exists())

    def test_switching_driver_does_not_reuse_the_other_modes_orbitals(self):
        # A common reference-free template makes mode the only changed input.
        shared = (EXAMPLES / 'orca_hf.template').read_text().replace('RHF ', '')
        (self.work / 'orca.template').write_text(shared)
        self.run_script()
        self.mode = 'dft'
        self.run_script(reset=False)
        self.assertNotIn('MORead', (self.work / 'orca_dft_polarized.inp').read_text())

    def test_invalid_pointcharge_count(self):
        (self.work / 'point charges').write_text('2\n0.5 0 0 5\n')
        self.run_script(ok=False)
        self.assertFalse((self.work / 'calls').exists())

    def test_missing_executable(self):
        self.env['ORCA_EXE'] = str(self.work / 'missing')
        self.run_script(ok=False)
        self.assertFalse((self.work / 'calls').exists())


if __name__ == '__main__':
    unittest.main()
