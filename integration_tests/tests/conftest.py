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

    tmpdir = "tmpdir"

    if os.path.exists(tmpdir) and os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)

    shutil.copytree(os.path.join("tests", example_dir), tmpdir)

    os.chdir(tmpdir)

    yield tmpdir

    os.chdir("..")
    shutil.rmtree(tmpdir)


def execute_pq(input_file):

    # Path to the pq executable
    folder_path = pathlib.Path(__file__).parent.absolute()

    # PQ lies in ../../build/apps/PQ

    pq_executable = os.path.join(
        folder_path, "..", "..", "build", "apps", "PQ")

    # Run the pq executable with the input file
    process = subprocess.Popen(
        [pq_executable, input_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return stdout, stderr
