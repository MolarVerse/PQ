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
import zipfile
import pytest


ZIP_CACHE_DIRNAME = ".zip_cache"


def _resolve_example_dir(file_folder, example_dir):
    """
    Resolves example_dir to a source directory to copy test data from.

    Large example data sets are stored as "<top-level-dir>.zip" next to
    their directory (e.g. "qm-mm.zip") to keep the repository small. If
    such a zip exists for the top-level component of example_dir, it is
    lazily extracted into a cache dir and the matching subdirectory of the
    extracted tree is returned. Otherwise example_dir is used as-is.
    """

    example_path = pathlib.Path(example_dir)
    top_level = example_path.parts[0]

    zip_path = file_folder / f"{top_level}.zip"
    if not zip_path.exists():
        return file_folder / example_path

    cache_dir = file_folder / ZIP_CACHE_DIRNAME
    if not (cache_dir / top_level).exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)

    return cache_dir / example_path


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

    source_dir = _resolve_example_dir(file_folder, example_dir)
    tmpdir = str(file_folder / "tmpdir")

    if os.path.exists(tmpdir) and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)

    print(os.path.abspath(tmpdir))

    shutil.copytree(source_dir, tmpdir)

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
        [pq_executable, input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    stdout = process.stdout
    stderr = process.stderr

    return stdout, stderr
