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

"""
This module contains fixtures and helper functions for the tests.
"""




import os
import shutil
import pathlib
import subprocess
import pytest
@pytest.fixture(scope="function")
def tmpdir():

    tmpdir = "tmpdir"

    if os.path.exists(tmpdir) and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.mkdir(tmpdir)

    os.chdir(tmpdir)

    yield tmpdir

    os.chdir("..")
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def test_with_data_dir(example_dir):

    file_folder = pathlib.Path(__file__).parent.absolute()

    example_dir = file_folder / example_dir
    tmpdir = str(file_folder / "tmpdir")

    if os.path.exists(tmpdir) and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)

    print(os.path.abspath(tmpdir))

    shutil.copytree(example_dir, tmpdir)

    os.chdir(tmpdir)

    yield tmpdir

    os.chdir("..")
    shutil.rmtree(tmpdir)


def execute_pq(input_file):

    # Path to the pq executable
    folder_path = pathlib.Path(__file__).parent.absolute()

    # PQ lies in ../../build/apps/PQ

    pq_executable = os.path.join(
        folder_path, "..", "build", "apps", "PQ")

    # Run the pq executable with the input file
    process = subprocess.run(
        [pq_executable, input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    print(stdout)
    print(stderr)

    return stdout, stderr
