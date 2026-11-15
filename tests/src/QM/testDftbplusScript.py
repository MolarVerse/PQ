"""Check the DFTB+ driver against the removed Perl driver; usage: <script> <tests/data/dftbplusScript>.

Each case holds the inputs, the files real DFTB+ 25.1 wrote (replayed by a
fake dftb+), and every file the Perl driver left behind for the same calls.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(sys.argv.pop(1)).resolve()
DATA = Path(sys.argv.pop(1)).resolve()
FAKE = r'''#!/usr/bin/env python3
import os, shutil
from pathlib import Path
store = Path(os.environ["FAKE_DFTB_DIR"])
counter = Path(".fake_dftb_calls")
index = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(index + 1))
shutil.copyfile("dftb_in.hsd", f"seen_dftb_in.{index}.hsd")
for path in sorted((store / str(index)).iterdir()):
    shutil.copyfile(path, path.name)
'''


class DftbplusScriptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pq-dftb-")
        self.root = Path(self.temp.name)
        fake = self.root / "bin" / "dftb+"
        fake.parent.mkdir()
        fake.write_text(FAKE)
        fake.chmod(0o755)
        self.path = f"{fake.parent}{os.pathsep}{os.environ['PATH']}"

    def tearDown(self):
        self.temp.cleanup()

    def test_matches_perl_byte_for_byte(self):
        cases = sorted(p for p in DATA.iterdir() if p.is_dir())
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case.name):
                work = self.root / case.name
                shutil.copytree(case / "inputs", work)
                env = dict(os.environ, PATH=self.path, FAKE_DFTB_DIR=str(case / "dftb"))
                done = subprocess.run(
                    [sys.executable, str(SCRIPT), *(case / "args").read_text().split()],
                    cwd=work, env=env, capture_output=True, check=False)
                self.assertEqual((case / "failed").read_text().strip() == "1", done.returncode != 0,
                                 done.stderr.decode(errors="replace"))
                want = {p.name: p.read_bytes() for p in (case / "expected").iterdir()}
                got = {p.name: p.read_bytes() for p in work.iterdir() if p.is_file()}
                self.assertEqual(sorted(want), sorted(got))
                for name, data in want.items():
                    self.assertEqual(data, got[name], name)

    def test_missing_dftb_output_fails(self):
        case = DATA / "qmmm"
        work = self.root / "broken"
        shutil.copytree(case / "inputs", work)
        empty = self.root / "empty" / "0"
        empty.mkdir(parents=True)
        env = dict(os.environ, PATH=self.path, FAKE_DFTB_DIR=str(empty.parent))
        done = subprocess.run([sys.executable, str(SCRIPT), *(case / "args").read_text().split()],
                              cwd=work, env=env, capture_output=True, check=False)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn(b"Error moving detailed.out", done.stderr)


if __name__ == "__main__":
    unittest.main()
