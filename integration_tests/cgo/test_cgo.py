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
