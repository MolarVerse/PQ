"""Tests of scripts/apt_install.sh with a fake sudo, apt-get, dpkg and apt directories."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "apt_install.sh"

FAKE_SUDO = '#!/bin/bash\n[ "$1" = --preserve-env ] && shift\nexec "$@"\n'

# What the n-th `apt-get install` does is read from line n of $FAKE_DIR/install.plan
# (the last line repeats): ok, fail, hang (no output, no progress), slow (writes
# data for 3 s, then succeeds), endless (writes data for ever).
FAKE_APT_GET = """#!/bin/bash
echo "apt-get $*" >> "$FAKE_DIR/calls"
for argument in "$@"; do
    if [ "$argument" = install ]; then
        n=$(grep -c " install " "$FAKE_DIR/calls")
        plan=$(sed -n "${n}p" "$FAKE_DIR/install.plan")
        [ -z "$plan" ] && plan=$(tail -n 1 "$FAKE_DIR/install.plan")
        case "$plan" in
            ok) exit 0 ;;
            fail) exit 100 ;;
            hang) sleep 30; exit 0 ;;
            slow) for i in 1 2 3 4 5 6 7 8 9 10; do echo "$i" >> "$FAKE_DIR/apt/archives/part"; sleep 0.3; done; exit 0 ;;
            endless) while true; do echo x >> "$FAKE_DIR/apt/archives/part"; sleep 0.2; done ;;
        esac
    fi
done
exit 0
"""

FAKE_DPKG = '#!/bin/bash\necho "dpkg $*" >> "$FAKE_DIR/calls"\nexit 0\n'


class AptInstallTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, text in (("sudo", FAKE_SUDO), ("apt-get", FAKE_APT_GET), ("dpkg", FAKE_DPKG)):
            path = self.bin / name
            path.write_text(text)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        (self.root / "apt" / "archives").mkdir(parents=True)
        (self.root / "calls").write_text("")

    def run_script(self, plan, *packages, **env):
        (self.root / "install.plan").write_text("\n".join(plan) + "\n")
        environment = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            FAKE_DIR=str(self.root),
            APT_PROGRESS_DIRS=str(self.root / "apt"),
            APT_STALL="1",
            APT_POLL="0.2",
            APT_RETRY_PAUSE="0",
        )
        environment.pop("APT_INSTALL_AS_ROOT", None)
        environment.update(env)
        result = subprocess.run(
            ["bash", str(SCRIPT), *(packages or ("gcc-14",))],
            capture_output=True, text=True, env=environment,
        )
        return result, (self.root / "calls").read_text().splitlines()

    @staticmethod
    def installs(calls, packages="gcc-14"):
        return sum(c.endswith(f"install -y {packages}") for c in calls)

    def test_a_working_install_runs_once(self):
        result, calls = self.run_script(["ok"], "gcc-14", "g++-14")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(2, len(calls))
        self.assertIn("update", calls[0])
        self.assertTrue(calls[1].endswith("install -y gcc-14 g++-14"))

    def test_the_apt_network_timeouts_and_retries_are_set(self):
        _, calls = self.run_script(["ok"])
        for option in ("Acquire::Retries=3", "Acquire::http::Timeout=20", "DPkg::Lock::Timeout=120"):
            self.assertIn(option, calls[0])
            self.assertIn(option, calls[1])

    def test_a_failed_try_is_repeated(self):
        result, calls = self.run_script(["fail", "ok"])
        self.assertEqual(0, result.returncode)
        self.assertEqual(2, self.installs(calls))
        self.assertIn("dpkg --configure -a", calls)
        self.assertIn("::warning::apt-get try 1 of 3", result.stdout)

    def test_a_stalled_try_is_stopped_and_repeated(self):
        result, calls = self.run_script(["hang", "ok"])
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(2, self.installs(calls))
        self.assertIn("stopping", result.stdout)

    def test_a_slow_install_that_still_makes_progress_is_not_stopped(self):
        # 3 s of writing against a stall limit of 1 s
        result, calls = self.run_script(["slow"])
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(1, self.installs(calls))
        self.assertNotIn("stopping", result.stdout)

    def test_an_install_that_runs_for_ever_is_stopped_by_the_cap(self):
        result, calls = self.run_script(["endless"], APT_CAP="1", APT_TRIES="2")
        self.assertEqual(1, result.returncode)
        self.assertEqual(2, self.installs(calls))
        self.assertIn("running 1", result.stdout)

    def test_it_gives_up_after_the_last_try(self):
        result, calls = self.run_script(["fail"], APT_TRIES="2")
        self.assertEqual(1, result.returncode)
        self.assertEqual(2, self.installs(calls))
        self.assertIn("::error::apt-get did not finish in 2 tries", result.stdout)

    def test_no_package_is_an_error(self):
        result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(2, result.returncode)


if __name__ == "__main__":
    unittest.main()
