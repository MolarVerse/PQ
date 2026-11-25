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
Cell list together with a manostat.

A manostat changes the box in every step, so the cell list has to rebuild its
cells while it is in use. Two tests cover this:

- the NPT input with the cell list must reproduce the stored brute-force NPT
  reference (the box hardly moves there: 0.01 % in 10 steps);
- the same input with a fast manostat (relaxation time 0.1 ps: the box moves
  between 26.6 and 31.4 Angstrom in 10 steps) must give the same result with
  and without the cell list. No reference data is needed: the two runs are
  compared with each other, the brute-force run is the reference.

The second test uses 16 cells per dimension. Rebuilding the cells matters only
when the box changes the number of neighbouring cells a pair search needs,
ceil(cutoff / cell size). With 16 cells the box of the start file needs 6, and
the box at its smallest needs 7: a cell list that keeps its first layout then
misses the pairs between 9.96 and 10 Angstrom, which the forces show at once.
(With the default of 7 cells the number stays 3 for every box of this run, a
stale layout would still give the right result, and the test could not tell.)
"""

import pathlib
import re

import numpy as np
import pytest

from conftest import execute_pq
from analysis import check_pq_output

FAST_MANOSTAT = 0.1  # ps, relaxation time of the Berendsen manostat
CELL_NUMBER = 16
CUTOFF = 10.0  # Angstrom, rcoulomb of run-01_NPT.in

# the cell list sums the pair interactions in a different order than the brute
# force loop, so the results agree to rounding (measured: positions and
# velocities identical at the printed 8 digits, forces 1e-11 and energies 3e-14
# relative); a missing neighbour cell changes the forces by 1e-3 and more
POSITION_TOLERANCE = 1e-7  # Angstrom
VELOCITY_TOLERANCE = 1e-7  # relative to the largest velocity
FORCE_TOLERANCE = 1e-8  # relative to the largest force
ENERGY_TOLERANCE = 1e-9  # relative
BOX_TOLERANCE = 1e-9  # relative


def _write_input(target, prefix, cell_list, relaxation_time=None,
                 every_step=False, cell_number=None):
    text = pathlib.Path("run-01_NPT.in").read_text()
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)

    if every_step:
        text = re.sub(r"output_freq\s*=\s*\d+", "output_freq = 1", text)

    if relaxation_time is not None:
        text = re.sub(r"p_relaxation\s*=\s*[0-9.]+",
                      f"p_relaxation = {relaxation_time}", text)

    if cell_list:
        number = f" cell-number = {cell_number};" if cell_number else ""
        text = text.replace(
            "integrator = v-verlet;",
            f"integrator = v-verlet; cell-list = on;{number}", 1)

    pathlib.Path(target).write_text(text)


def _run(prefix, cell_list, relaxation_time=None, every_step=False,
         cell_number=None):
    _write_input(f"{prefix}.in", prefix, cell_list, relaxation_time,
                 every_step, cell_number)

    _, stderr = execute_pq(f"{prefix}.in")

    assert stderr == b""


def _frames(path):
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


def _energies(path):
    """All columns of the energy file except the last two (momentum, time)."""
    return np.array([[float(x) for x in line.split()[:-2]]
                     for line in pathlib.Path(path).read_text().splitlines()
                     if line.strip() and not line.startswith("#")])


START_FILE = "cgo-mm-00.rst"


def _read_restart_box(path):
    """First box length of a restart file ("Step ..." is line 1, "Box ..." 2,
    after the optional "!META_DATA" lines)."""
    lines = [
        line for line in pathlib.Path(path).read_text().splitlines()
        if not line.startswith("!META_DATA")
    ]
    line = lines[1]
    assert line.startswith("Box")
    return float(line.split()[1])


def _boxes(path):
    return np.array([[float(x) for x in line.split()[1:4]]
                     for line in pathlib.Path(path).read_text().splitlines()])


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_cgo_NPT_cellList(test_with_data_dir):
    _run("cgo-mm-01", cell_list=True)

    check_pq_output("cgo-mm-01", "cgo", "ref_data_NPT")


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_cgo_NPT_cellList_follows_a_moving_box(test_with_data_dir):
    _run("brute_force", cell_list=False, relaxation_time=FAST_MANOSTAT,
         every_step=True)
    _run("cell_list", cell_list=True, relaxation_time=FAST_MANOSTAT,
         every_step=True, cell_number=CELL_NUMBER)

    # the test is only meaningful if the layout of the cells had to change: the
    # smallest box of the run needs more neighbouring cells than the start box
    start_box = _read_restart_box(START_FILE)
    boxes = _boxes("cell_list.box")

    def neighbour_cells(length):
        return int(np.ceil(CUTOFF * CELL_NUMBER / length))

    assert 27.0 < start_box < 28.0
    assert neighbour_cells(start_box) == 6
    assert neighbour_cells(boxes[:, 0].min()) == 7
    assert 2 * neighbour_cells(boxes[:, 0].min()) + 1 <= CELL_NUMBER

    # a frame of every step, so every rebuild of the cells is compared
    for kind, tolerance in (("xyz", POSITION_TOLERANCE),
                            ("vel", VELOCITY_TOLERANCE),
                            ("force", FORCE_TOLERANCE)):
        reference = _frames(f"brute_force.{kind}")
        result = _frames(f"cell_list.{kind}")

        assert len(result) == len(reference) == 10

        scale = 1.0 if kind == "xyz" else max(np.abs(f).max()
                                              for f in reference)

        for frame, frame_reference in zip(result, reference):
            assert np.abs(frame - frame_reference).max() <= tolerance * scale

    assert np.allclose(_boxes("cell_list.box"), _boxes("brute_force.box"),
                       rtol=BOX_TOLERANCE, atol=0)

    reference = _energies("brute_force.en")
    result = _energies("cell_list.en")

    assert result.shape == reference.shape
    assert np.allclose(result, reference, rtol=ENERGY_TOLERANCE, atol=0)
