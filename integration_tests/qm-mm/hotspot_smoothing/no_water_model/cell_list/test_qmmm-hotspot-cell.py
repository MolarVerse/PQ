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


FORCE_RTOL = 1e-6
FORCE_ATOL = 1e-8


@pytest.mark.parametrize(
    "example_dir",
    ["qm-mm/hotspot_smoothing/no_water_model/cell_list/"],
    indirect=False,
)
def test_qmmm_hotspot_cell(test_with_data_dir):
    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01.in")

    assert stderr == b""

    check_pq_output(
        "output-md-01",
        "qm-mm/hotspot_smoothing/no_water_model/cell_list",
        force_rtol=FORCE_RTOL,
        force_atol=FORCE_ATOL,
    )
