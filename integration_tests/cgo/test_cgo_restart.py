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
PQ wrote must give the same result as one run that is not stopped.

The restart file holds positions with 8 decimals and velocities with 9
significant digits, so the two runs agree to that precision, not bit for bit.
The tolerances are a few times the rounding that the stiff Buckingham forces
amplify (measured: positions 1e-8, velocities 8e-8 and forces 2e-7 relative to
their largest value); a restart that changes the box or loses a part of the
state is off by 1e-4 to 1e-3.
"""

import pathlib
import re

import numpy as np
import pytest

from conftest import execute_pq

START_FILE = "cgo-mm-00.rst"
N_STEPS_FIRST = 5
N_STEPS_SECOND = 5

POSITION_TOLERANCE = 1e-7  # Angstrom
VELOCITY_TOLERANCE = 1e-6  # relative to the largest velocity
FORCE_TOLERANCE = 1e-5  # relative to the largest force
BOX_TOLERANCE = 1e-9  # relative


def _write_input(source, target, n_steps, prefix, start_file):
    text = pathlib.Path(source).read_text()
    text = re.sub(r"nstep\s*=\s*\d+", f"nstep = {n_steps}", text)
    text = re.sub(r"output_freq\s*=\s*\d+", "output_freq = 1", text)
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)
    text = re.sub(r"start_file\s*=\s*[^;]+", f"start_file = {start_file}", text)
    pathlib.Path(target).write_text(text)


def _run(source, n_steps, prefix, start_file):
    target = f"{prefix}.in"
    _write_input(source, target, n_steps, prefix, start_file)

    _, stderr = execute_pq(target)

    assert stderr == b""


def _read_frames(path):
    """Positions, velocities or forces of every frame of a trajectory file."""
    lines = pathlib.Path(path).read_text().splitlines()
    frames = []
    index = 0

    while index < len(lines):
        n_atoms = int(lines[index].split()[0])
        block = lines[index + 2:index + 2 + n_atoms]
        frames.append(np.array([[float(x) for x in line.split()[1:4]]
                                for line in block]))
        index += n_atoms + 2

    return frames


def _read_restart(path):
    """Step, box lengths and (position, velocity, force) of a restart file."""
    lines = pathlib.Path(path).read_text().splitlines()
    step = int(lines[0].split()[1])
    box = np.array([float(x) for x in lines[1].split()[1:4]])
    atoms = np.array([[float(x) for x in line.split()[3:12]]
                      for line in lines[2:] if len(line.split()) >= 12])

    return step, box, atoms


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
@pytest.mark.parametrize("input_file", ["run-01.in", "run-01_NPT.in"])
def test_cgo_restart_round_trip(test_with_data_dir, input_file):
    n_steps = N_STEPS_FIRST + N_STEPS_SECOND

    _run(input_file, n_steps, "full", START_FILE)
    _run(input_file, N_STEPS_FIRST, "first", START_FILE)
    _run(input_file, N_STEPS_SECOND, "second", "first.rst")

    # the first half of the stopped run is the same simulation as the start of
    # the continuous one, bit for bit
    for kind in ("xyz", "vel", "force"):
        full = _read_frames(f"full.{kind}")
        first = _read_frames(f"first.{kind}")

        assert len(first) == N_STEPS_FIRST
        assert all(np.array_equal(a, b)
                   for a, b in zip(first, full[:N_STEPS_FIRST]))

    # the step counter continues instead of starting again
    start_step, start_box, _ = _read_restart(START_FILE)
    first_step, first_box, _ = _read_restart("first.rst")
    full_step, full_box, full_atoms = _read_restart("full.rst")
    second_step, second_box, second_atoms = _read_restart("second.rst")

    assert first_step == start_step + N_STEPS_FIRST
    assert second_step == full_step == start_step + n_steps

    # the box of the restart file is the box of the run, not a shortened
    # text of it (the NVE box does not change at all)
    if input_file == "run-01.in":
        assert np.array_equal(first_box, start_box)
        assert np.array_equal(second_box, start_box)

    assert np.allclose(second_box, full_box, rtol=BOX_TOLERANCE, atol=0)

    # the continued run follows the continuous one frame by frame
    for kind, tolerance in (("xyz", POSITION_TOLERANCE),
                            ("vel", VELOCITY_TOLERANCE),
                            ("force", FORCE_TOLERANCE)):
        full = _read_frames(f"full.{kind}")[N_STEPS_FIRST:]
        second = _read_frames(f"second.{kind}")

        assert len(second) == N_STEPS_SECOND

        scale = 1.0 if kind == "xyz" else max(np.abs(f).max() for f in full)

        for frame_full, frame_second in zip(full, second):
            assert np.abs(frame_full - frame_second).max() <= tolerance * scale

    # and ends in the same restart file
    assert np.abs(full_atoms[:, 0:3] - second_atoms[:, 0:3]).max() \
        <= POSITION_TOLERANCE
