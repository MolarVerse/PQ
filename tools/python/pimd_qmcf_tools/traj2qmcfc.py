"""
*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

import numpy as np
import sys

from .common import print_header


def print_help():

    print_header()

    print("Usage: python xyz2qmcfc.py [options] file1 file2 ...")
    print("Options with arbitrary position:")
    print("  -h, --help    Show this help message")


def combine_input_files():

    combined_files = ""
    number_of_files = 0

    while len(sys.argv) > 1:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print_help()
            sys.exit()
        else:
            combined_files += open(sys.argv[1]).read()
            sys.argv.pop(1)
            number_of_files += 1

    return combined_files, number_of_files


def main():

    combined_files, number_of_files = combine_input_files()

    combined_files_lines = combined_files.split("\n")

    i = 0

    while i < len(combined_files_lines) - 1:

        line = combined_files_lines[i]
        components = line.split()

        n_atoms = int(components[0])
        x, y, z, alpha, beta, gamma = components[1:7]

        i += 2
        atoms = []

        # Loop over the remaining lines in the file
        for j in range(i, i + n_atoms):
            atoms.append(combined_files_lines[j])

            i += 1

        print(f"{n_atoms+1} {x} {y} {z} {alpha} {beta} {gamma}")
        print("")
        print(f"X   0.0 0.0 0.0")
        for j in range(n_atoms):
            print(atoms[j])
