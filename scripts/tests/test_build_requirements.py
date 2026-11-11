"""The stated compiler and CMake requirements agree with what the build enforces.

external/mstd is part of every build and hard-fails below its limits, so the
root CMakeLists.txt, the README, the installation page, AGENTS.md and the
container and conda scripts have to state the same numbers (#1107).
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return (ROOT / path).read_text()


def cmake_minimum(path):
    return re.search(r"cmake_minimum_required\(VERSION (\d+\.\d+)", read(path)).group(1)


def gcc_minimum(path):
    return re.search(r"GNU.*?VERSION_LESS ([\d]+)\.0", read(path), re.DOTALL).group(1)


class BuildRequirementsTest(unittest.TestCase):
    def setUp(self):
        self.cmake = cmake_minimum("CMakeLists.txt")
        self.gcc = gcc_minimum("CMakeLists.txt")

    def test_the_root_is_at_least_as_strict_as_the_vendored_mstd(self):
        def as_tuple(version):
            return tuple(int(part) for part in version.split("."))

        self.assertGreaterEqual(as_tuple(self.cmake), as_tuple(cmake_minimum("external/mstd/CMakeLists.txt")))
        self.assertGreaterEqual(int(self.gcc), int(gcc_minimum("external/mstd/CMakeLists.txt")))

    def test_the_error_message_names_the_checked_gcc_version(self):
        self.assertIn(f"need >= {self.gcc}.0", read("CMakeLists.txt"))

    def test_the_documents_state_the_same_cmake_and_gcc(self):
        for path in ("README.md", "AGENTS.md", "docs/sphinx/src/installation/installation.rst"):
            with self.subTest(path=path):
                text = read(path)
                self.assertRegex(text, rf"CMake >= {re.escape(self.cmake)}\b")
                self.assertRegex(text, rf"GCC\s+>= {self.gcc}\.0")

    def test_the_container_recipes_install_that_gcc(self):
        for path in ("scripts/PQ.def", "scripts/PQ_openmpi.def"):
            with self.subTest(path=path):
                text = read(path)
                self.assertIn(f"install gcc-{self.gcc}", text)
                self.assertIn(f"install g++-{self.gcc}", text)
                self.assertNotRegex(text, rf"gcc-(?!{self.gcc}\b)\d+")

    def test_the_conda_script_checks_that_gcc(self):
        text = read("scripts/conda_build.sh")
        self.assertIn(f'-lt {self.gcc} ]', text)
        self.assertIn(f"at least {self.gcc}", text)

    def test_the_ci_installs_a_gcc_that_is_allowed(self):
        default = re.search(r"version:\n(?:.*\n)*?\s+default: '(\d+)'", read(".github/actions/setup-gcc/action.yml"))
        self.assertGreaterEqual(int(default.group(1)), int(self.gcc))


if __name__ == "__main__":
    unittest.main()
