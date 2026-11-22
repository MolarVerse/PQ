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

The values of the MM Hessian, which the `oh_mm_hessian` tests only check for
shape and finiteness (the Hessian of that example is the zero matrix: its two
atoms do not interact).

One O-H bond with a harmonic potential E = k / 2 (r - r0)^2 (the bonded force
field, k = 500 kcal/mol/Angstrom^2, r0 = 1 Angstrom, no other interaction) has
an analytic Hessian. For the vector d from atom 1 to atom 2, r = |d| and
u = d / r the 3 x 3 block of atom 1 is

    k (u u^T + (1 - r0 / r) (1 - u u^T))

the block of atom 2 is the same, the two off-diagonal blocks are its negative.
The bond is tilted against the axes, so every element is different, and it is
stretched (r = 1.65) or at its equilibrium length (r = r0) where the Hessian
has rank 1.

The numerical builders differ in the order of their truncation error, which
the tests check: central differences (h^2), forward (h) and five point (h^4).
"""

import math
import pathlib
import re
import subprocess

import pytest

from conftest import execute_pq

FORCE_CONSTANT = 500.0  # kcal/mol/Angstrom^2
EQUILIBRIUM = 1.0  # Angstrom
GEOMETRIES = {
    "stretched": ("start.rst", (-0.3, -0.4, -0.5), (0.5, 0.4, 0.7)),
    "equilibrium": ("start_equilibrium.rst", (-0.3, -0.4, -0.5),
                    (0.3, -0.4, 0.3)),
}
# measured for the stretched bond at a displacement of 0.001 Angstrom:
# central 4.3e-5, forward 0.10, five point 5.8e-11
MAX_ERROR = {"central": 5e-4, "forward": 1.0, "five_point": 1e-7}
MIN_ERROR = {"central": 1e-6, "forward": 1e-2, "five_point": 0.0}


def analytic_hessian(first, second):
    """The 6 x 6 Hessian of the harmonic bond between two positions."""
    vector = [b - a for a, b in zip(first, second)]
    distance = math.sqrt(sum(value * value for value in vector))
    unit = [value / distance for value in vector]
    scale = 1.0 - EQUILIBRIUM / distance

    block = [[FORCE_CONSTANT * (unit[i] * unit[j]
                                + scale * ((i == j) - unit[i] * unit[j]))
              for j in range(3)] for i in range(3)]

    return [[(1 if row // 3 == column // 3 else -1) * block[row % 3][column % 3]
             for column in range(6)] for row in range(6)]


def run_hessian(builder, geometry, displacement=0.001, prefix="bond"):
    """Run PQ and return the Hessian as a list of rows."""
    start_file = GEOMETRIES[geometry][0]
    text = pathlib.Path("run.in").read_text()
    text = re.sub(r"start_file = .*;", f"start_file = {start_file};", text)
    text = re.sub(r"hessian_builder = .*;", f"hessian_builder = {builder};",
                  text)
    text = re.sub(r"hessian_displacement = .*;",
                  f"hessian_displacement = {displacement};", text)
    text = text.replace("bond.hessian", f"{prefix}.hessian")
    text = text.replace("file_prefix = bond;", f"file_prefix = {prefix};")
    pathlib.Path(f"{prefix}.in").write_text(text)

    _, stderr = execute_pq(f"{prefix}.in")

    assert stderr == b""

    rows = [[float(value) for value in line.split()]
            for line in pathlib.Path(f"{prefix}.hessian").read_text()
            .splitlines() if line.strip()]
    assert len(rows) == 6 and all(len(row) == 6 for row in rows)

    return rows


def largest_difference(first, second):
    return max(abs(a - b) for row_a, row_b in zip(first, second)
               for a, b in zip(row_a, row_b))


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("builder", ["central", "forward", "five_point"])
def test_numerical_hessian_equals_the_analytic_one(
        test_with_data_dir, builder, geometry):
    _, first, second = GEOMETRIES[geometry]
    expected = analytic_hessian(first, second)

    hessian = run_hessian(builder, geometry)

    error = largest_difference(hessian, expected)
    assert error <= MAX_ERROR[builder], (
        f"{builder} Hessian of the {geometry} bond is off by {error:.2e}")

    # the order of the truncation error is a property of the builder; a builder
    # that silently used another stencil would fall out of its range
    if geometry == "stretched":
        assert error >= MIN_ERROR[builder], (
            f"{builder} is more accurate than its stencil allows ({error:.2e})")

    # the Hessian is symmetric, and a shift of both atoms costs no energy
    for row in range(6):
        for column in range(6):
            assert hessian[row][column] == pytest.approx(
                hessian[column][row], abs=1e-9)
    if builder != "forward":
        for row in range(6):
            for axis in range(3):
                assert hessian[row][axis] + hessian[row][3 + axis] == \
                    pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_central_hessian_error_follows_the_displacement(test_with_data_dir):
    _, first, second = GEOMETRIES["stretched"]
    expected = analytic_hessian(first, second)

    small = largest_difference(
        run_hessian("central", "stretched", 0.001, "small"), expected)
    large = largest_difference(
        run_hessian("central", "stretched", 0.002, "large"), expected)

    # central differences: the error grows with the square of the displacement
    assert 3.0 < large / small < 5.0, (
        f"errors {small:.2e} and {large:.2e} do not scale as h squared")


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_hessian_info_file_describes_the_run(test_with_data_dir):
    run_hessian("five_point", "stretched")

    info = pathlib.Path("bond.hessian.info").read_text()

    assert "format = pq-hessian-info-v1" in info
    assert "hessian_builder = FIVE_POINT" in info
    assert "hessian_displacement = 0.001" in info
    assert "hessian_definition = -dF_i/dx_j" in info
    assert "hessian_unit = kcal_mol-1_angstrom-2" in info
    assert "rows = 6" in info and "columns = 6" in info


@pytest.mark.parametrize("example_dir", ["hessian_bond/"], indirect=False)
def test_analytic_builder_is_refused_for_the_mm_evaluator(test_with_data_dir):
    text = pathlib.Path("run.in").read_text().replace(
        "hessian_builder = central;", "hessian_builder = analytic;")
    pathlib.Path("analytic.in").write_text(text)

    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_pq("analytic.in")

    assert error.value.returncode != 0
    output = error.value.stdout + error.value.stderr
    assert b"does not support analytic Hessians" in output
