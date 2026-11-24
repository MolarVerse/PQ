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

The cell list together with the force field, the topology, the intra
non-bonded file, SHAKE and Wolf: the chignolin in water (7382 atoms). The other
cell list tests use no force field.

There is no reference. The same run with and without the cell list must agree
(a missing neighbour cell changes the forces by 1e-3 and more, the different
order of summation only by rounding). The cutoff is 10 Angstrom instead of the
20 of the reference run: the cell list needs a box of at least twice the cutoff
and the box of the chignolin is 40 to 43 Angstrom wide.
"""

import pathlib
import re

import numpy as np
import pytest

from conftest import execute_pq

CUTOFF = 10.0  # Angstrom

# measured: positions identical, velocities 1e-9 and forces 1e-10 relative to
# the largest value, energies below 1e-12 relative
POSITION_TOLERANCE = 1e-7  # Angstrom
VELOCITY_TOLERANCE = 1e-7  # relative to the largest velocity
FORCE_TOLERANCE = 1e-8  # relative to the largest force
ENERGY_TOLERANCE = 1e-9  # relative


def _run(prefix, cell_list, cell_number=None):
    text = pathlib.Path("run-01.in").read_text()
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)
    text = re.sub(r"rcoulomb\s*=\s*[0-9.]+", f"rcoulomb = {CUTOFF}", text)
    text = re.sub(r"nstep\s*=\s*\d+", "nstep = 5", text)

    if cell_list:
        number = f" cell-number = {cell_number};" if cell_number else ""
        text = text.replace(
            "integrator = v-verlet;",
            f"integrator = v-verlet; cell-list = on;{number}", 1)

    pathlib.Path(f"{prefix}.in").write_text(text)

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


@pytest.mark.parametrize("example_dir", ["chig/"], indirect=False)
@pytest.mark.parametrize("cell_number", [None, 9], ids=["7_cells", "9_cells"])
def test_chig_cell_list_equals_brute_force(test_with_data_dir, cell_number):
    _run("brute_force", cell_list=False)
    _run("cell_list", cell_list=True, cell_number=cell_number)

    for kind, tolerance in (("xyz", POSITION_TOLERANCE),
                            ("vel", VELOCITY_TOLERANCE),
                            ("force", FORCE_TOLERANCE)):
        reference = _frames(f"brute_force.{kind}")
        result = _frames(f"cell_list.{kind}")

        assert len(result) == len(reference) == 5

        scale = 1.0 if kind == "xyz" else max(np.abs(f).max()
                                              for f in reference)

        for frame, frame_reference in zip(result, reference):
            assert frame.shape == frame_reference.shape
            assert np.abs(frame - frame_reference).max() <= tolerance * scale, (
                f"the {kind} file of the cell list differs from brute force")

    reference = _energies("brute_force.en")
    result = _energies("cell_list.en")

    assert result.shape == reference.shape
    assert np.allclose(result, reference, rtol=ENERGY_TOLERANCE, atol=0)
