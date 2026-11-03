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

import math
import pathlib
import subprocess

import pytest

from conftest import execute_pq

# The shipped example is the test data: it is an MM-only workflow (no QM
# program needed) and testing it also keeps the example itself working.
EXAMPLE_DIR = "../examples/oh_mm_hessian"
ATOMS = 2
SIZE = 3 * ATOMS


def read_hessian(prefix):
    """The Hessian as a list of rows; fails if it is not a finite SIZE x SIZE matrix."""
    rows = [
        [float(value) for value in line.split()]
        for line in pathlib.Path(f"{prefix}.hessian").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == SIZE
    for row in rows:
        assert len(row) == SIZE
        assert all(math.isfinite(value) for value in row)
    return rows


def check_hessian_info(prefix):
    info = pathlib.Path(f"{prefix}.hessian.info").read_text()
    assert "format = pq-hessian-info-v1" in info
    assert f"rows = {SIZE}" in info
    assert f"columns = {SIZE}" in info


def run_pq(input_file):
    """execute_pq, but a failing PQ reports what it printed instead of only its exit status."""
    try:
        return execute_pq(input_file)
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"PQ {input_file} exited with {error.returncode}\n"
            f"stderr: {error.stderr.decode(errors='replace')}\n"
            f"stdout (end): {error.stdout.decode(errors='replace')[-1500:]}"
        )


def run_and_check(input_file, prefix):
    stdout, stderr = run_pq(input_file)

    assert stderr == b""
    assert b"PQ ended normally" in stdout

    read_hessian(prefix)
    check_hessian_info(prefix)


def with_extra_keywords(input_file, keywords):
    """Copy of the input with additional keywords (the original stays untouched)."""
    path = pathlib.Path(input_file)
    modified = path.with_name(f"modified-{path.name}")
    modified.write_text(path.read_text() + "\n" + keywords + "\n")
    return modified.name


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_optimization_and_hessian_without_max_learning_rate(test_with_data_dir):
    # max-learning-rate is optional (no default); an input without it used to
    # stop with "Setting value is not set and no default is available."
    assert "max-learning-rate" not in pathlib.Path("run-01.in").read_text()

    run_and_check("run-01.in", "oh-opt")

    # the optimization really ran: it wrote its trajectory of steps
    assert pathlib.Path("oh-opt.opt").read_text().strip() != ""


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_hessian_without_optimization(test_with_data_dir):
    run_and_check("run-02-no-opt.in", "oh-no-opt")


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_optimization_and_hessian_with_max_learning_rate(test_with_data_dir):
    input_file = with_extra_keywords("run-01.in", "max-learning-rate = 1.0;")

    run_and_check(input_file, "oh-opt")


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_minimum_not_below_maximum_learning_rate_is_reported(test_with_data_dir):
    input_file = with_extra_keywords("run-01.in", "max-learning-rate = 1.0e-20;")

    with pytest.raises(subprocess.CalledProcessError) as error:
        execute_pq(input_file)

    assert error.value.returncode != 0
    assert b"The minimum learning rate" in error.value.stderr
    assert b"maximum learning rate" in error.value.stderr
