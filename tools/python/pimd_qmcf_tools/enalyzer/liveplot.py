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

import matplotlib.pyplot as plt
import numpy as np
import time
import os

from pimd_qmcf_tools.enalyzer.reader import read_en

# Define a function to read the data from the file and update the plot
def update_plot(en_filenames, selected):
    
    # Read the data from the file
    data = read_en(en_filenames)

    looptime = data[-1].iloc[-1,-1]
    
    plt.clf()

    for df in data:
        x = df.get(0)
        y = df.get(selected.get())

        plt.plot(x, y)
    
    plt.draw()
    plt.pause(looptime)

    return looptime

def liveplot(en_filenames, selected):
    # Set the filename to monitor

    # Initialize the plot
    plt.ion()

    # Initial plot setup and looptime
    looptime = update_plot(en_filenames, selected)

    list_modified_time = []

    # set last modified time
    for filename in en_filenames:
        list_modified_time.append(os.path.getmtime(filename))

    # Wait for changes to the file
    try:
        while len(plt.get_fignums()) != 0:
            # Check if the file has been modified
            for (i, filename) in enumerate(en_filenames):
                modified_time = os.path.getmtime(filename)
                if modified_time != list_modified_time[i]:
                    # Set the new modified time
                    list_modified_time[i] = modified_time
                    # Update the plot
                    looptime = update_plot(en_filenames, selected)

            # Pause for 1 second
            time.sleep(looptime)
    except KeyboardInterrupt:
        pass