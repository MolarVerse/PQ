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

import pytest
import os

from conftest import execute_pq
from analysis import check_pq_output


@pytest.mark.parametrize(
    "example_dir",
    ["cgo/"],
    indirect=False
)
def test_cgo(test_with_data_dir):
    # print list of all files in current directory and path of current directory

    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01.in")

    # check if stderr is empty
    assert stderr == b""

    check_pq_output("cgo-mm-01", "cgo")


@pytest.mark.parametrize(
    "example_dir",
    ["cgo/"],
    indirect=False
)
def test_cgo_NPT(test_with_data_dir):
    # print list of all files in current directory and path of current directory

    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01_NPT.in")

    # check if stderr is empty
    assert stderr == b""

    check_pq_output("cgo-mm-01", "cgo", "ref_data_NPT")
