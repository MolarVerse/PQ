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

import os

import pytest

from analysis import check_pq_output
from conftest import execute_pq

torch = pytest.importorskip("torch")
pytest.importorskip("mace")

# PQ runs MACE on a CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MACE needs a CUDA device"
)


@pytest.mark.parametrize(
    "example_dir",
    ["mace/"],
    indirect=False
)
def test_mace(test_with_data_dir):
    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01.in")

    assert stderr == b""

    check_pq_output("malondialdehyde-md-01", "mace")
