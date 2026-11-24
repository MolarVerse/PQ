"""
*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
*****************************************************************************

A start file without atoms is refused for every job type (issue 1238). Before,
mm-md ran with all zeros, mm-opt crashed (segmentation fault), the Nose-Hoover
chain wrote NaN into the restart file and the thermostats stopped with a
message about a zero-temperature system.
"""

import pathlib
import subprocess

import pytest

from conftest import execute_pq

EXAMPLE_DIR = "../examples/oh_mm_hessian"
MESSAGE = (b'Start file "no_atoms.rst" contains no atoms - a simulation needs '
           b'at least one atom')

COMMON = """force-field = off;
rcoulomb = 9.0;
start_file = no_atoms.rst;
file_prefix = empty;
"""
INPUTS = {
    "mm-md": "jobtype = mm-md;\nnstep = 5;\ntimestep = 1.0;\n" + COMMON,
    "mm-md-nose-hoover": (
        "jobtype = mm-md;\nnstep = 5;\ntimestep = 1.0;\n"
        "thermostat = nh-chain; temp = 300.0; nh-chain_length = 3;\n"
        "coupling_frequency = 3000;\n" + COMMON),
    "mm-opt": (
        "jobtype = mm-opt;\nnstep = 3;\noptimizer = steepest-descent;\n"
        "learning-rate-strategy = constant;\ninitial-learning-rate = 0.001;\n"
        "energy-conv-strategy = absolute;\nabs-energy-conv = 1.0e10;\n"
        "rel-energy-conv = 1.0e10;\nmax-force-conv = 1.0e10;\n"
        "rms-force-conv = 1.0e10;\n" + COMMON),
    "mm-hessian": (
        "jobtype = mm-hessian;\nhessian_file = empty.hessian;\n"
        "hessian_info_file = empty.hessian.info;\n"
        "hessian_displacement = 0.001;\noptimize_before_hessian = false;\n"
        "hessian_builder = central;\n" + COMMON),
}


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
@pytest.mark.parametrize("job", sorted(INPUTS))
@pytest.mark.parametrize(
    "start_file", ["Step 0\nBox 20.0 20.0 20.0\n", "", "# nothing\n"],
    ids=["box_only", "empty_file", "only_a_comment"])
def test_a_start_file_without_atoms_is_refused(
        test_with_data_dir, job, start_file):
    pathlib.Path("no_atoms.rst").write_text(start_file)
    pathlib.Path("run.in").write_text(INPUTS[job])

    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_pq("run.in")

    assert error.value.returncode == 1, "a clear error, not a crash"
    assert MESSAGE in error.value.stdout + error.value.stderr


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_the_start_file_with_atoms_is_still_accepted(test_with_data_dir):
    text = pathlib.Path("run-02-no-opt.in").read_text()

    _, stderr = execute_pq("run-02-no-opt.in")

    assert stderr == b""
    assert "start.rst" in text
