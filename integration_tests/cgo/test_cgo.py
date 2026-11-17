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
def test_cgo_cellList(test_with_data_dir):
    # print list of all files in current directory and path of current directory

    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01_celllist.in")

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


@pytest.mark.parametrize(
    "example_dir",
    ["cgo/"],
    indirect=False
)
def test_cgo_atomic_virial(test_with_data_dir):
    # print list of all files in current directory and path of current directory

    print("Current directory: ", os.getcwd())
    print("List of files in current directory: ", os.listdir(os.getcwd()))

    stdout, stderr = execute_pq("run-01_atomic_virial.in")

    # check if stderr is empty
    assert stderr == b""

    check_pq_output("cgo-mm-01", "cgo", "ref_data_atomic_virial")


@pytest.mark.parametrize(
    "example_dir",
    ["cgo/"],
    indirect=False
)
def test_cgo_temperature_ramp(test_with_data_dir):
    # Berendsen thermostat with a temperature ramp from 200 K to 998.15 K over
    # 8 steps in 4 updates; the trajectory differs from every other cgo test
    # by the ramp only, so the other tests keep their coverage untouched

    stdout, stderr = execute_pq("run-01_temp_ramp.in")

    # check if stderr is empty
    assert stderr == b""

    check_pq_output("cgo-mm-01", "cgo", "ref_data_temp_ramp")

    # the ramp setup is reported in the log file: (998.15 - 200) / 4 updates
    with open("cgo-mm-01.log") as log:
        log_text = log.read()

    assert "Start temp:          200 K" in log_text
    assert "Temp ramp increase:  199.5375 K" in log_text
    assert "Temp ramp steps:     8" in log_text
    assert "Temp ramp frequency: 2" in log_text

