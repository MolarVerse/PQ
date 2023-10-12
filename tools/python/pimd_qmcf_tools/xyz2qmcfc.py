import numpy as np
import sys

from .common import print_header


def print_help():

    print_header()

    print("Usage: python xyz2qmcfc.py [options] file1 file2 ...")
    print("Options with arbitrary position:")
    print("  -h, --help    Show this help message")


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

combined_files_lines = combined_files.split("\n")

for i in range(len(combined_files_lines)):

    components = combined_files_lines[i].split()

    n_atoms = int(components[0])
    x, y, z, alpha, beta, gamma = map(
        float, components[1:7])

    i += 1
    names = []
    coordinates = []

    # Loop over the remaining lines in the file
    for j in range(i, i + n_atoms):
        # Split the line into its components
        components = combined_files_lines[j].split()

        # Extract the coordinates and convert them to floats
        name = components[0]
        x, y, z = map(float, components[1:4])

        # Append the coordinates to the list
        names.append(name)
        coordinates.append([x, y, z])

        i += 1

# Convert the list of coordinates to a NumPy array
coordinates = np.array(coordinates)

# Print the number of atoms and the coordinates
print(f"Number of atoms: {n_atoms}")
print("Coordinates:")
print(coordinates)
