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

Triclinic simulation boxes. They are only supported for QM-MD and ring polymer
QM-MD (issue 910 tracks MM-MD), so these tests use dftb+ with the 9-atom
malondialdehyde of the dftbplus test, in a box small enough (9 x 10 x 11
Angstrom, angles 75, 80, 70 degrees) that the periodic images interact and a
wrong lattice would change the energy.

* NVE: compared with a reference, and the lattice that PQ hands to dftb+ is
  checked on its own (lengths and angles of the three vectors).
* Berendsen manostat with `isotropy = full_anisotropic`: the box angles move.
  Compared with a reference.
* The same with `isotropy = anisotropic`: the angles must stay exactly as they
  are (no reference).
* `full_anisotropic` also works from a box with 90 degree angles (it has to be
  a box that can shear, otherwise the molecules are sheared in a cell that is
  not: issue 1297).
* An MM-MD run with a triclinic start box, or with `full_anisotropic`, is
  refused with a clear message.
"""

import math
import pathlib
import re
import subprocess

import pytest

from analysis import DFTB_FORCE_ATOL, check_pq_output
from conftest import execute_pq

PREFIX = "malondialdehyde-md-01"
LENGTHS = (9.0, 10.0, 11.0)
ANGLES = (75.0, 80.0, 70.0)


def _assert_finite(*extensions):
    for extension in extensions:
        text = pathlib.Path(f"{PREFIX}.{extension}").read_text().lower()
        assert "nan" not in text, f"{PREFIX}.{extension} contains NaN"


def _boxes():
    """The six box parameters of every frame of the .box file."""
    return [
        [float(value) for value in line.split()[1:7]]
        for line in pathlib.Path(f"{PREFIX}.box").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _lattice_parameters(path):
    """Lengths and angles (alpha between b and c, beta between a and c, gamma
    between a and b) of the three lattice vectors of a dftb+ gen file: its last
    four lines are the origin and the vectors."""
    lines = [
        line for line in pathlib.Path(path).read_text().splitlines()
        if line.strip()
    ]
    a, b, c = [[float(value) for value in line.split()]
               for line in lines[-3:]]

    def length(vector):
        return math.sqrt(sum(value * value for value in vector))

    def angle(first, second):
        cosine = sum(x * y for x, y in zip(first, second)) / (
            length(first) * length(second))
        return math.degrees(math.acos(cosine))

    return ((length(a), length(b), length(c)),
            (angle(b, c), angle(a, c), angle(a, b)))


@pytest.mark.parametrize("example_dir", ["dftbplus/"], indirect=False)
def test_dftbplus_triclinic_box(test_with_data_dir):
    _, stderr = execute_pq("run-01_triclinic.in")

    assert stderr == b""

    _assert_finite("xyz", "vel", "force", "en")

    lengths, angles = _lattice_parameters("geom.gen")
    assert lengths == pytest.approx(LENGTHS, abs=1e-6)
    assert angles == pytest.approx(ANGLES, abs=1e-6)

    check_pq_output(PREFIX, "dftbplus", "ref_data_triclinic",
                    force_atol=DFTB_FORCE_ATOL)


@pytest.mark.parametrize("example_dir", ["dftbplus/"], indirect=False)
def test_dftbplus_triclinic_box_full_anisotropic_manostat(test_with_data_dir):
    _, stderr = execute_pq("run-01_triclinic_npt.in")

    assert stderr == b""

    _assert_finite("xyz", "vel", "force", "en", "box")

    boxes = _boxes()
    assert len(boxes) > 1
    changes = [abs(boxes[-1][3 + index] - expected)
               for index, expected in enumerate(ANGLES)]
    assert min(changes) > 1e-3 and max(changes) > 0.1, (
        f"the box angles must change with full_anisotropic: {changes}")
    for index, expected in enumerate(LENGTHS):
        assert abs(boxes[-1][index] - expected) > 1e-3

    check_pq_output(PREFIX, "dftbplus", "ref_data_triclinic_npt",
                    force_atol=DFTB_FORCE_ATOL)


@pytest.mark.parametrize("example_dir", ["dftbplus/"], indirect=False)
def test_dftbplus_triclinic_box_anisotropic_manostat_keeps_the_angles(
        test_with_data_dir):
    text = pathlib.Path("run-01_triclinic_npt.in").read_text()
    assert "isotropy = full_anisotropic;" in text
    pathlib.Path("anisotropic.in").write_text(
        text.replace("isotropy = full_anisotropic;", "isotropy = anisotropic;"))

    _, stderr = execute_pq("anisotropic.in")

    assert stderr == b""

    _assert_finite("xyz", "vel", "force", "en", "box")

    boxes = _boxes()
    assert len(boxes) > 1
    for box in boxes:
        assert box[3:] == pytest.approx(ANGLES, abs=1e-9)
    assert abs(boxes[-1][0] - LENGTHS[0]) > 1e-3, "the lengths must move"


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_mm_md_refuses_a_triclinic_box(test_with_data_dir):
    """Issue 910: classical MM-MD cannot use a triclinic box yet. When it can,
    this test has to be replaced by a real one."""
    start = pathlib.Path("cgo-mm-00.rst")
    text, count = re.subn(
        r"^Box\s+(\S+)\s+(\S+)\s+(\S+).*$", r"Box \1 \2 \3 75.0 80.0 70.0",
        start.read_text(), count=1, flags=re.MULTILINE)
    assert count == 1
    start.write_text(text)

    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_pq("run-01.in")

    assert error.value.returncode != 0
    output = error.value.stdout + error.value.stderr
    assert b"Triclinic box is only supported for QM-MD and RP-QM-MD" in output


@pytest.mark.parametrize("example_dir", ["dftbplus/"], indirect=False)
def test_dftbplus_full_anisotropic_manostat_from_a_box_with_right_angles(
        test_with_data_dir):
    """The start box has no angles; full_anisotropic has to be able to change
    them (issue 1297: before, the box stayed orthorhombic)."""
    start = pathlib.Path("input.rst")
    text, count = re.subn(r"^Box\s+.*$", "Box   9.0   10.0   11.0",
                          start.read_text(), count=1, flags=re.MULTILINE)
    assert count == 1
    pathlib.Path("cubic.rst").write_text(text)

    text = pathlib.Path("run-01_triclinic_npt.in").read_text()
    assert "input_triclinic.rst" in text
    pathlib.Path("cubic.in").write_text(
        text.replace("input_triclinic.rst", "cubic.rst"))

    _, stderr = execute_pq("cubic.in")

    assert stderr == b""

    _assert_finite("xyz", "vel", "force", "en", "box")

    boxes = _boxes()
    assert len(boxes) > 1
    changes = [abs(angle - 90.0) for angle in boxes[-1][3:]]
    assert min(changes) > 1e-3 and max(changes) > 0.05, (
        f"the box angles must change from 90 degrees: {changes}")
    assert abs(boxes[-1][0] - LENGTHS[0]) > 1e-3, "the lengths must move"


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_mm_md_refuses_a_full_anisotropic_manostat(test_with_data_dir):
    """Issue 1297 (and 910): without a triclinic box in MM-MD the shear of a
    full anisotropic manostat made the box explode; now it is refused."""
    text = pathlib.Path("run-01_NPT_anisotropic.in").read_text()
    assert "isotropy = anisotropic;" in text
    pathlib.Path("full.in").write_text(
        text.replace("isotropy = anisotropic;", "isotropy = full_anisotropic;"))

    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_pq("full.in")

    assert error.value.returncode != 0
    output = error.value.stdout + error.value.stderr
    assert b"isotropy = full_anisotropic needs a triclinic box" in output
