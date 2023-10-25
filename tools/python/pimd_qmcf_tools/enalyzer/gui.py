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
from customtkinter import *
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

# Add the path to the read_en module and feature module
from pimd_qmcf_tools.enalyzer.reader import read_en, read_info
from pimd_qmcf_tools.enalyzer.plot import live_graph, graph

# Create the root window
window = CTk()
window._set_appearance_mode("light")

def gui(en_filenames, info_filename):

    window.title('Energy File Analyzer - PIMD-QMCF')

    # Read all en files
    data = read_en(en_filenames)
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
        radio = CTkRadioButton(window, text=text, value=j, variable=selected)
        radio.grid(column=j % 2, row=int(j/2))

    # Call select features
    list_features_select = select_features()

    def graph_wrapper():
        list_features = get_features(list_features_select)
        graph(data=data, info_list=info_list, en_filenames=en_filenames, selected=selected, list_features=list_features)

    def live_graph_wrapper():
        live_graph(en_filenames=en_filenames, selected=selected)

    button = CTkButton(window, text="Graph It!", command=graph_wrapper)
    button.grid(column=3, row=j)

    button_lp = CTkButton(window, text="Live Graph!", command=live_graph_wrapper)
    button_lp.grid(column=4, row = j)

    def statistics_window_wrapper():
        statistics_window(data=data, en_filenames=en_filenames, selected=selected)

    CTkButton(window, text="Click to open statistical information", command=statistics_window_wrapper).grid(row=int((j+1)/2))

    window.mainloop()

    return None


def select_features():
    row_counter = -1

    row_counter = row_counter+1
    CTkLabel(window, text="Time Axis:").grid(row = row_counter, column=3)

    time_step = StringVar()
    row_counter = row_counter+1
    CTkLabel(window, text="Time step (fs):").grid(row = row_counter, column=3)
    CTkEntry(window, textvariable=time_step).grid(row=row_counter, column=4)

    row_counter = row_counter+1
    CTkLabel(window, text="Analysis Tools:").grid(row=row_counter, column=3)

    running_average_kernel = StringVar()
    row_counter = row_counter+1
    CTkLabel(window, text="Window Size:").grid(row=row_counter, column=3)
    CTkEntry(window, textvariable=running_average_kernel).grid(row=row_counter, column=4)

    running_average = IntVar()
    row_counter = row_counter+1
    Checkbutton(window, text="Running Average", indicator=0, variable=running_average).grid(row=row_counter, column=3)

    overall_mean = IntVar()
    row_counter = row_counter+1
    Checkbutton(window, text="Overall Mean", indicator=0, variable=overall_mean).grid(row=row_counter, column=3)

    integrate = IntVar()
    row_counter = row_counter+1
    Checkbutton(window, text="Integratation", indicator=0, variable=integrate).grid(row=row_counter, column=3)

    integration_average = IntVar()
    row_counter = row_counter+1
    Checkbutton(window, text="Integratation Average", indicator=0, variable=integration_average).grid(row=row_counter, column=3)

    return [time_step, running_average_kernel, running_average, overall_mean, integrate, integration_average]


def get_features(list_features_select):
    time_step = list_features_select[0]
    running_average_kernel = list_features_select[1]
    running_average = list_features_select[2]
    overall_mean = list_features_select[3]
    integrate = list_features_select[4]

    return [time_step.get(), running_average_kernel.get(), running_average.get(), overall_mean.get(), integrate.get()]

def statistics_window(data, en_filenames, selected):
    mean = []
    std_dev = []
    total_datasets = []
    for (i, data_frame) in enumerate(data):
        y = data_frame.get(selected.get())
        mean.append(np.mean(y))
        std_dev.append(np.std(y))
        total_datasets.append(y)

    new_window = Toplevel(window)
    new_window.title("Statistical information")
    Text(window)

    # calculate means and standard deviations
    for (i, average) in enumerate(mean):
        message = "Mean of "+str(en_filenames[i])+" = "+str(
            round(average, 3))+", Standard deviation = "+str(round(std_dev[i], 3))
        Label(new_window, text=message).pack()

    total_datasets = np.concatenate(total_datasets)
    total_mean = np.mean(total_datasets)
    total_std_dev = np.std(total_datasets)
    conclusion_message = "Mean over all datasets= " + \
        str(round(total_mean, 3))+", Standard deviation = " + \
        str(round(total_std_dev, 3))
    Label(new_window, text=conclusion_message).pack()