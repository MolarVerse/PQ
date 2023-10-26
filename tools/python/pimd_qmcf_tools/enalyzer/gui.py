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
from tkinter import IntVar, StringVar, Text
import customtkinter as ctk
import numpy as np

# Add the path to the read_en module and feature module
from pimd_qmcf_tools.enalyzer.reader import read_en, read_info
from pimd_qmcf_tools.enalyzer.plot import live_graph, graph

def gui(en_filenames, info_filename):

    # Set the appearance of the window
    ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
    ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

    # Create the root window
    window = ctk.CTk()

    # Set the title of the window
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
        radio = ctk.CTkRadioButton(window, text=text, value=j, variable=selected)
        radio.grid(column=j % 2, row=int(j/2), sticky="W", padx=10, pady=5)

    # Call select features
    list_features_select, row_counter = select_features(window)

    def graph_wrapper():
        list_features = get_features(list_features_select)
        graph(data=data, info_list=info_list, en_filenames=en_filenames, selected=selected, list_features=list_features)

    button = ctk.CTkButton(window, text="Graph It!", command=graph_wrapper)
    button.grid(column=3, row=row_counter+1, columnspan=2, padx=10, pady=5)

    def live_graph_wrapper():
        live_graph(en_filenames=en_filenames, selected=selected)

    button_lp = ctk.CTkButton(window, text="Live Graph!", command=live_graph_wrapper)
    button_lp.grid(column=3, row = row_counter+2, columnspan=2, padx=20, pady=5)

    def statistics_window_wrapper():
        statistics_window(window=window, data=data, en_filenames=en_filenames, selected=selected)

    button_stat = ctk.CTkButton(window, text="Click to open statistical information", command=statistics_window_wrapper)
    button_stat.grid(row=int((j+1)/2), columnspan=2, padx=20, pady=5)

    window.mainloop()

    return None


def select_features(window):
    row_counter = -1

    row_counter = row_counter+1
    ctk.CTkLabel(window, text="Time Axis:").grid(row = row_counter, column=3, columnspan=2, pady=5)

    time_step = StringVar()
    row_counter = row_counter+1
    ctk.CTkLabel(window, text="Time step (fs):").grid(row = row_counter, column=3)
    ctk.CTkEntry(window, textvariable=time_step).grid(row=row_counter, column=4)

    row_counter = row_counter+1
    ctk.CTkLabel(window, text="Analysis Tools:").grid(row=row_counter, column=3, columnspan=2, pady=5)

    running_average_kernel = StringVar()
    row_counter = row_counter+1
    ctk.CTkLabel(window, text="Window Size:").grid(row=row_counter, column=3)
    ctk.CTkEntry(window, textvariable=running_average_kernel).grid(row=row_counter, column=4)

    running_average = IntVar()
    row_counter = row_counter+1
    ctk.CTkCheckBox(window, text="Running Average", variable=running_average).grid(row=row_counter, column=3, columnspan=2, sticky="W")

    overall_mean = IntVar()
    row_counter = row_counter+1
    ctk.CTkCheckBox(window, text="Overall Mean", variable=overall_mean).grid(row=row_counter, column=3, columnspan=2, sticky="W")

    integrate = IntVar()
    row_counter = row_counter+1
    ctk.CTkCheckBox(window, text="Integratation", variable=integrate).grid(row=row_counter, column=3, columnspan=2, sticky="W")

    integration_average = IntVar()
    row_counter = row_counter+1
    ctk.CTkCheckBox(window, text="Integratation Average", variable=integration_average).grid(row=row_counter, column=3, columnspan=2, sticky="W")

    return [time_step, running_average_kernel, running_average, overall_mean, integrate, integration_average], row_counter


def get_features(list_features_select):
    time_step = list_features_select[0]
    running_average_kernel = list_features_select[1]
    running_average = list_features_select[2]
    overall_mean = list_features_select[3]
    integrate = list_features_select[4]

    return [time_step.get(), running_average_kernel.get(), running_average.get(), overall_mean.get(), integrate.get()]

def statistics_window(window, data, en_filenames, selected):
    mean = []
    std_dev = []
    total_datasets = []
    for (i, data_frame) in enumerate(data):
        y = data_frame.get(selected.get())
        mean.append(np.mean(y))
        std_dev.append(np.std(y))
        total_datasets.append(y)

    new_window = ctk.CTkToplevel(window)
    new_window.title("Statistical information")
    Text(window)

    # calculate means and standard deviations
    for (i, average) in enumerate(mean):
        message = "Mean of "+str(en_filenames[i])+" = "+str(
            round(average, 3))+", Standard deviation = "+str(round(std_dev[i], 3))
        ctk.CTkLabel(new_window, text=message).pack(fill=ctk.BOTH)

    total_datasets = np.concatenate(total_datasets)
    total_mean = np.mean(total_datasets)
    total_std_dev = np.std(total_datasets)
    conclusion_message = "Mean over all datasets = " + \
        str(round(total_mean, 3))+", Standard deviation = " + \
        str(round(total_std_dev, 3))
    ctk.CTkLabel(new_window, text=conclusion_message).pack(fill=ctk.BOTH)