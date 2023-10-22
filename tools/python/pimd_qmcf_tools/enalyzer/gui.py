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
from tkinter import *
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

# Add the path to the read_en module and feature module
from pimd_qmcf_tools.enalyzer.reader import read_info, read_en
from pimd_qmcf_tools.enalyzer.feature import get_feature
from pimd_qmcf_tools.enalyzer.liveplot import liveplot

# Signal handler for closing the gui window
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    root.destroy()
    sys.exit(0)

# Register signal handler with SIGINT signal
signal.signal(signal.SIGINT, signal_handler)

# Create the root window
root = Tk()

def gui(en_filenames, info_filename):

    root.config(bg="white")
    root.title('Energy File Analyzer - PIMD-QMCF')

    # Read all en files
    data = read_en(en_filenames)

    # Read info file
    info_list = read_info(info_filename)

    # check if same number of columns are present in .en file as in .info suggested
    for i in range(len(data)):
        if len(data[i].axes[1]) != len(info_list):
            print("Info file does not match " +
                  en_filenames[i]+" energy file! (Unequal number of columns)")
            exit(1)

    # Draw RadioButtons from info list
    selected = IntVar()

    for (j, text) in enumerate(info_list):
        radio = Radiobutton(root, text=text, value=j, indicator=0, variable=selected,
                            background="#d4d700", font='sans 14')
        radio.grid(column=j % 2, row=int(j/2),
                   ipadx=10, ipady=5, sticky=W+E+N+S)

    # Call select features
    select_features()

    # Open matplotlib window
    def graph():
        for (i, dataframe) in enumerate(data):
            
            list_features = get_features()

            # Checks if time step is given and converts to ps
            if not (list_features[0] == ""):
                timeStepValue = float(list_features[0])
                x = dataframe.get(0).apply(lambda x: x * timeStepValue / 1000)
            else:
                x = dataframe.get(0)
            
            y = dataframe.get(selected.get())
            label = str(info_list[selected.get()]) + \
                " (" + str(en_filenames[i]) + ")"
            plt.plot(x, y, label=label)

            kernelSize = 100
            if runningAverageKernel.get() != "":
                kernelSize = int(runningAverageKernel.get())

            if any(list_features):
                for (i, value) in enumerate(list_features):
                    if (value == 1):
                        (f, label) = get_feature(i, value, y, kernelSize)
                        plt.plot(x, f, label=label)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=2, fancybox=True, shadow=True)
        
        # Checks if time step is given
        if not (list_features[0] == ""):
            plt.xlabel("Simulation Time in ps")
        else:
            plt.xlabel("Frame")

        plt.show()

    def liveplot_wrapper():
        liveplot(en_filenames=en_filenames, selected=selected)

    button = Button(root, text="Graph It!", command=graph,
                    background="#007f5f", font='sans 16 bold')
    button.grid(column=3, row=j , ipadx=15, ipady=15)

    button_lp = Button(root, text="Live Plot!", command=liveplot_wrapper,
                    background="#007f5f", font='sans 16 bold')
    button_lp.grid(column=4, row = j, ipadx=15, ipady=15)

    def statistics_window():
        mean = []
        std_dev = []
        total_datasets = []
        for (i, dataframe) in enumerate(data):
            y = dataframe.get(selected.get())
            mean.append(np.mean(y))
            std_dev.append(np.std(y))
            total_datasets.append(y)

        newWindow = Toplevel(root)
        newWindow.title("Statistical informations")
        T = Text(root)

        # calculate means and standard deviations
        for (i, average) in enumerate(mean):
            message = "Mean of "+str(en_filenames[i])+" = "+str(
                round(average, 3))+", Standard deviation = "+str(round(std_dev[i], 3))
            Label(newWindow, text=message, font='sans 12 ').pack()

        total_datasets = np.concatenate(total_datasets)
        total_mean = np.mean(total_datasets)
        total_std_dev = np.std(total_datasets)
        conclusion_message = "Mean over all datasets= " + \
            str(round(total_mean, 3))+", Standard deviation = " + \
            str(round(total_std_dev, 3))
        Label(newWindow, text=conclusion_message, font='sans 12 bold').pack()

    Button(root, text="Click to open statistical informations", command=statistics_window,
           background="#80b918", font='sans 14').grid(columnspan=2, row=int((j+1)/2), ipadx=10, ipady=5, sticky=W+E+N+S)

    root.mainloop()

    return None


def select_features():
    global timeStep
    global runningAverage
    global runningAverageKernel
    global overallMean
    global integrate
    global integratationAverage

    rowCounter = -1

    Label(root, text="Time Axis:", font='sans 16 bold',
        background="white").grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2)

    timeStep = StringVar()
    Label(root, text="Time step (fs):", background='white', font='sans 14').grid(
        row=(rowCounter := rowCounter+1), column=3, ipadx=5, ipady=5, sticky=W+E+N+S)
    Entry(root, textvariable=timeStep, font='sans 14').grid(
        row=rowCounter, column=4, ipadx=5, sticky=W+E+N+S)

    Label(root, text="Analysis Tools:", font='sans 16 bold',
          background="white").grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2)

    runningAverageKernel = StringVar()
    Label(root, text="Window Size:", background='white', font='sans 14').grid(
        row=(rowCounter := rowCounter+1), column=3, ipadx=5, ipady=5, sticky=W+E+N+S)
    Entry(root, textvariable=runningAverageKernel, font='sans 14').grid(
        row=rowCounter, column=4, ipadx=5, sticky=W+E+N+S)

    runningAverage = IntVar()
    Checkbutton(root, text="Running Average", indicator=0, variable=runningAverage,
                background="#55a630", font='sans 14').grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2, ipadx=5, ipady=5, sticky=W+E+N+S)

    overallMean = IntVar()
    Checkbutton(root, text="Overall Mean", indicator=0, variable=overallMean,
                background="#55a630", font='sans 14').grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2, ipadx=5, ipady=5, sticky=W+E+N+S)

    integrate = IntVar()
    Checkbutton(root, text="Integratation", indicator=0, variable=integrate,
                background="#55a630", font='sans 14').grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2, ipadx=5, ipady=5, sticky=W+E+N+S)

    integratationAverage = IntVar()
    Checkbutton(root, text="Integratation Average", indicator=0, variable=integratationAverage,
                background="#55a630", font='sans 14').grid(row=(rowCounter := rowCounter+1), column=3, columnspan=2, ipadx=5, ipady=5, sticky=W+E+N+S)

    return None


def get_features():
    return [timeStep.get(), runningAverage.get(), overallMean.get(), integrate.get(), integratationAverage.get()]