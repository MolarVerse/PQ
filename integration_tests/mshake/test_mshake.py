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

M-SHAKE on rigid TIP3P water: 2000 molecules (6000 atoms) in a box of 39.4
Angstrom, the `h2o_mm-mshake` example, 10 steps with a Berendsen thermostat
(the example's velocity rescaling thermostat is stochastic).

Besides the comparison with a reference the test checks the constraints
themselves, which does not depend on the reference: every water keeps its O-H
and H-H distances of the geometry in `mshake.top` in every frame, and the
relative velocity of two atoms of a water has no component along their bond.
"""

import math
import pathlib

import pytest

from analysis import check_pq_output
from conftest import execute_pq

PREFIX = "h2o-md-01"

# geometry of mshake.top: O at the origin, H at (0.585882, +-0.756950, 0)
OH_DISTANCE = math.hypot(0.585882, 0.756950)
HH_DISTANCE = 2 * 0.756950
DISTANCE_TOLERANCE = 1e-6   # Angstrom; M-SHAKE reaches about 2e-8
VELOCITY_TOLERANCE = 1e-5   # relative velocity along a bond / its size
BONDS = ((0, 1, OH_DISTANCE), (0, 2, OH_DISTANCE), (1, 2, HH_DISTANCE))


def _frames(path):
    """The box lengths and the three coordinates of every atom of every frame
    of an xyz or vel file."""
    lines = pathlib.Path(path).read_text().split("\n")
    frames = []
    index = 0

    while index < len(lines) and lines[index].strip():
        header = lines[index].split()
        count = int(header[0])
        box = [float(value) for value in header[1:4]]
        atoms = [
            (line.split()[0], [float(value) for value in line.split()[1:4]])
            for line in lines[index + 2:index + 2 + count]
        ]
        frames.append((box, atoms))
        index += count + 2

    return frames


def _minimum_image(vector, box):
    return [value - length * round(value / length)
            for value, length in zip(vector, box)]


def _length(vector):
    return math.sqrt(sum(value * value for value in vector))


def _waters(atoms):
    """The (O, H, H) triples of a frame; the names are checked."""
    assert len(atoms) % 3 == 0
    for first in range(0, len(atoms), 3):
        assert [atoms[first + offset][0] for offset in range(3)] == [
            "O", "H", "H"]
        yield first


@pytest.mark.parametrize("example_dir", ["mshake/"], indirect=False)
def test_mshake_water(test_with_data_dir):
    _, stderr = execute_pq("run-01.in")

    assert stderr == b""

    for extension in ("xyz", "vel", "force", "en"):
        text = pathlib.Path(f"{PREFIX}.{extension}").read_text().lower()
        assert "nan" not in text, f"{PREFIX}.{extension} contains NaN"

    positions = _frames(f"{PREFIX}.xyz")
    velocities = _frames(f"{PREFIX}.vel")
    assert len(positions) == len(velocities) == 2

    worst_distance = 0.0
    worst_velocity = 0.0
    for (box, atoms), (_, speeds) in zip(positions, velocities):
        for first in _waters(atoms):
            for one, two, expected in BONDS:
                bond = _minimum_image(
                    [b - a for a, b in zip(atoms[first + one][1],
                                           atoms[first + two][1])], box)
                worst_distance = max(
                    worst_distance, abs(_length(bond) - expected))

                relative = [b - a for a, b in zip(speeds[first + one][1],
                                                  speeds[first + two][1])]
                along = abs(sum(x * y for x, y in zip(bond, relative))
                            ) / _length(bond)
                worst_velocity = max(worst_velocity,
                                     along / max(_length(relative), 1e-300))

    assert worst_distance < DISTANCE_TOLERANCE, (
        f"a bond of a water is off by {worst_distance:.2e} Angstrom")
    assert worst_velocity < VELOCITY_TOLERANCE, (
        f"the relative velocity along a bond is {worst_velocity:.2e} of "
        "its size")

    check_pq_output(PREFIX, "mshake")
