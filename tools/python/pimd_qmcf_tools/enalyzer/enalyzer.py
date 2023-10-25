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

import os
import sys

# Add the path to the gui module
from pimd_qmcf_tools.enalyzer.gui import gui

def main():
    # create list of energy files and set info-exists checker to False
    en_filenames = []
    check_info_exists = False

    # Read all command line argument vectors
    for file in sys.argv[1:]:

        # Check if file ends in *.info and if a info-file was already selected
        if os.path.splitext(file)[-1] == ".info" and not check_info_exists:
            info_filename = file
            check_info_exists = True
        # Check if file ends in *.en
        elif os.path.splitext(file)[-1] == ".en":
            en_filenames.append(file)

    # Selects a info-file with the same filename as an en-file
    if not check_info_exists:
        for file in en_filenames:
            info_filename = os.path.splitext(file)[0] + ".info"
            if os.path.exists(info_filename):
                check_info_exists = True
                break

    # Checks if all en-files exist
    for file in en_filenames:
        if not os.path.exists(file):
            print("File: " + file + " does not exist!")
            exit(1)

    # Checks if the selected info-file exists.
    if not os.path.exists(info_filename):
        print("File: " + info_filename + " does not exist!")
        exit(1)


    # Opens the gui window
    gui(en_filenames=en_filenames, info_filename=info_filename)