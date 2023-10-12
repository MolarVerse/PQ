import math
import sys
import numpy as np
import re

from .common import print_header


def print_help():

    print_header()

    print("Usage: python box.py [options] file1 file2 ...")
    print("Options with arbitrary position:")
    print("  -h, --help    Show this help message")
    print("  --vmd         Output in VMD format")


def main():
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

    line_number = 1
    pi = math.pi

    for line in combined_files.split("\n"):
        nAtoms, x, y, z, alpha, beta, gamma = map(float, line.split())
        if not vmd_flag:
            print(line_number, x, y, z)
        else:
            print("8")
            print(f"Box {line_number} {x} {y} {z}")

            matrix = np.array([[x, y * math.cos(gamma * pi / 180), z*math.cos(beta * pi / 180)], [0, y * math.sin(gamma * pi / 180), z * (math.cos(alpha * pi / 180) - math.cos(beta * pi / 180) * math.cos(gamma * pi / 180)) /
                                                                                                  math.sin(gamma * pi / 180)], [0, 0, z * math.sqrt(1 - math.cos(beta * pi / 180)**2 - (math.cos(alpha * pi / 180) - math.cos(beta * pi / 180) * math.cos(gamma * pi / 180))**2 / math.sin(gamma * pi / 180)**2)]])

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

        line_number += 1
