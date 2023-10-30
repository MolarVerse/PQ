
"""
*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper, Josef M. Gallmetzer

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
import time
import os

from pimd_qmcf_tools.enalyzer.reader import read_en
from pimd_qmcf_tools.enalyzer.feature import get_feature

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

def live_graph(en_filenames, selected):
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

# Open matplotlib window
def graph(data, info_list, en_filenames, selected, list_features):
    for (i, data_frame) in enumerate(data):
        
        # Checks if time step is given and converts to ps
        if list_features[0] != "":
            time_step_value = float(list_features[0])
            x = data_frame.get(0).apply(lambda x, time=time_step_value: x * time / 1000)
        else:
            x = data_frame.get(0)
        
        y = data_frame.get(selected.get())
        label = str(info_list[selected.get()]) + \
            " (" + str(en_filenames[i]) + ")"
        plt.plot(x, y, label=label)

        kernel_size = 100
        running_average_kernel = list_features[1]
        if running_average_kernel != "":
            kernel_size = int(running_average_kernel)

        list_features = list_features[1:]

        if any(list_features):
            for (i, value) in enumerate(list_features):
                if (value == 1):
                    (f, label) = get_feature(i, value, y, kernel_size)
                    plt.plot(x, f, label=label)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)
    
    # Checks if time step is given
    if list_features[0] != "":
        plt.xlabel("Simulation Time in ps")
    else:
        plt.xlabel("Frame")

    plt.show()
