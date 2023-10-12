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

import math
import sys
import numpy as np
import re
from math import pi

from .common import print_header


def print_help():

    print_header()

    print("Usage: python box.py [options] file1 file2 ...")
    print("Options with arbitrary position:")
    print("  -h, --help    Show this help message")
    print("  --vmd         Output in VMD format")


def combine_input_files():
    combined_files = ""
    vmd_flag = False

    while len(sys.argv) > 1:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print_help()
            sys.exit()
        elif sys.argv[1] == "--vmd":
            vmd_flag = True
            sys.argv.pop(1)
        else:
            combined_files += open(sys.argv[1]).read()
            sys.argv.pop(1)

    if combined_files == "":
        print_help()
        sys.exit()

    combined_files = "\n".join(
        filter(lambda x: re.match("^[0-9]", x), combined_files.split("\n")))

    return combined_files, vmd_flag


def setup_box_matrix(x, y, z, alpha, beta, gamma):

    matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    matrix[0][0] = x
    matrix[0][1] = y * math.cos(gamma * pi / 180)
    matrix[0][2] = z * math.cos(beta * pi / 180)
    matrix[1][1] = y * math.sin(gamma * pi / 180)
    matrix[1][2] = z * (math.cos(alpha * pi / 180) - math.cos(beta * pi / 180)
                        * math.cos(gamma * pi / 180)) / math.sin(gamma * pi / 180)
    matrix[2][2] = z * math.sqrt(1 - math.cos(beta * pi / 180)**2 - (math.cos(alpha * pi / 180) - math.cos(
        beta * pi / 180) * math.cos(gamma * pi / 180))**2 / math.sin(gamma * pi / 180)**2)

    return matrix


def print_vmd_box(line_number, x, y, z, alpha, beta, gamma):
    print("8")
    print(f"Box {line_number} {x} {y} {z}")

    matrix = setup_box_matrix(x, y, z, alpha, beta, gamma)

    vec = matrix @ np.array([0.5, 0.5, 0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([0.5, 0.5, -0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([0.5, -0.5, 0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([0.5, -0.5, -0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([-0.5, 0.5, 0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([-0.5, 0.5, -0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([-0.5, -0.5, 0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")
    vec = matrix @ np.array([-0.5, -0.5, -0.5])
    print(f"X {vec[0]} {vec[1]} {vec[2]}")


def main():
    combined_files, vmd_flag = combine_input_files()

    line_number = 1
    pi = math.pi

    for line in combined_files.split("\n"):
        nAtoms, x, y, z, alpha, beta, gamma = map(float, line.split())
        if not vmd_flag:
            print(line_number, x, y, z)
        else:
            print_vmd_box(line_number, x, y, z, alpha, beta, gamma)

        line_number += 1
