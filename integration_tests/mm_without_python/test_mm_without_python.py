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
import pathlib
import re

import pytest

from conftest import execute_pq

EXAMPLE_DIR = "../examples/h2o_mm"


@pytest.mark.parametrize("example_dir", [EXAMPLE_DIR], indirect=False)
def test_mm_run_without_python_on_path_is_silent(test_with_data_dir):
    # an MM job must not start the embedded Python (#1264)
    input_file = pathlib.Path("run-01.in")
    input_file.write_text(
        re.sub(r"nstep\s*=\s*\d+", "nstep = 5", input_file.read_text()))

    empty_path = pathlib.Path("empty-path").absolute()
    empty_path.mkdir()
    env = dict(os.environ, PATH=str(empty_path))

    stdout, stderr = execute_pq(input_file.name, env=env)

    assert stderr == b""
    assert b"PQ ended normally" in stdout
