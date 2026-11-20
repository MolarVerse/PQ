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
"""


"""
Restart round trip: a run that is stopped and continued from the restart file
PQ wrote must give exactly the result of one run that is not stopped.

The restart file holds every value with the digits that read back as the same
number, so the continued run is not only close to the continuous one: its final
restart file is identical to the byte, and so are the last frames of the
trajectory, velocity and force files. The three variants cover the state of
the integrator (NVE), of the box (Berendsen manostat) and of a thermostat with
its own state (Nose-Hoover chain).
"""

import pathlib
import re

import numpy as np
import pytest

from conftest import execute_pq

START_FILE = "cgo-mm-00.rst"
N_STEPS_FIRST = 5
N_STEPS_SECOND = 5

NOSE_HOOVER = (
    "thermostat = nh-chain; temp = 998.15; nh-chain_length = 3; "
    "coupling_frequency = 3000;"
)


def _write_input(source, target, n_steps, prefix, start_file, extra):
    text = pathlib.Path(source).read_text()
    text = re.sub(r"nstep\s*=\s*\d+", f"nstep = {n_steps}", text)
    text = re.sub(r"output_freq\s*=\s*\d+", "output_freq = 1", text)
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)
    text = re.sub(r"start_file\s*=\s*[^;]+", f"start_file = {start_file}", text)
    text = text.replace("integrator = v-verlet;",
                        f"integrator = v-verlet; {extra}", 1)
    pathlib.Path(target).write_text(text)


def _run(source, n_steps, prefix, start_file, extra):
    target = f"{prefix}.in"
    _write_input(source, target, n_steps, prefix, start_file, extra)

    _, stderr = execute_pq(target)

    assert stderr == b""


def _frames(path):
    """The lines of every frame of a trajectory file."""
    lines = pathlib.Path(path).read_text().splitlines()
    frames = []
    index = 0

    while index < len(lines):
        n_atoms = int(lines[index].split()[0])
        frames.append(lines[index:index + n_atoms + 2])
        index += n_atoms + 2

    return frames


def _read_restart(path):
    """Step, box lengths and the number of chain levels of a restart file."""
    lines = pathlib.Path(path).read_text().splitlines()
    step = int(lines[0].split()[1])
    box = np.array([float(x) for x in lines[1].split()[1:4]])
    chain_levels = sum(line.startswith("chi") for line in lines)

    return step, box, chain_levels


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
@pytest.mark.parametrize(
    "input_file, extra",
    [
        pytest.param("run-01.in", "", id="NVE"),
        pytest.param("run-01_NPT.in", "", id="Berendsen_NPT"),
        pytest.param("run-01.in", NOSE_HOOVER, id="Nose_Hoover_chain"),
    ],
)
def test_cgo_restart_round_trip(test_with_data_dir, input_file, extra):
    n_steps = N_STEPS_FIRST + N_STEPS_SECOND

    _run(input_file, n_steps, "full", START_FILE, extra)
    _run(input_file, N_STEPS_FIRST, "first", START_FILE, extra)
    _run(input_file, N_STEPS_SECOND, "second", "first.rst", extra)

    # the first half of the stopped run is the same simulation as the start
    # of the continuous one
    for kind in ("xyz", "vel", "force"):
        full = _frames(f"full.{kind}")
        first = _frames(f"first.{kind}")

        assert len(first) == N_STEPS_FIRST
        assert first == full[:N_STEPS_FIRST]

    # the step counter continues instead of starting again
    start_step, start_box, _ = _read_restart(START_FILE)
    first_step, first_box, _ = _read_restart("first.rst")
    full_step, full_box, full_levels = _read_restart("full.rst")
    second_step, second_box, second_levels = _read_restart("second.rst")

    assert first_step == start_step + N_STEPS_FIRST
    assert second_step == full_step == start_step + n_steps

    # the box of the restart file is the box of the run, to the last digit:
    # an NVE box does not change at all
    if input_file == "run-01.in":
        assert np.array_equal(first_box, start_box)
        assert np.array_equal(second_box, start_box)

    # the Nose-Hoover chain is part of the restart file
    assert second_levels == full_levels == (3 if extra else 0)

    # the continued run follows the continuous one, frame by frame ...
    for kind in ("xyz", "vel", "force"):
        full = _frames(f"full.{kind}")[N_STEPS_FIRST:]
        second = _frames(f"second.{kind}")

        assert len(second) == N_STEPS_SECOND
        assert second == full

    # ... and ends in the same state, with every digit of the restart file
    assert pathlib.Path("second.rst").read_bytes() == \
        pathlib.Path("full.rst").read_bytes()
