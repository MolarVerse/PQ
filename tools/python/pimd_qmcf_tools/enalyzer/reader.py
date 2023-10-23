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

import pandas as pd

def read_en(energy_filenames):

    data = []
    for file in energy_filenames:
        data.append(pd.read_csv(file, delim_whitespace=True, header=None, comment='#'))

    return data

def read_info(info_filename):

    # read info, start from 4th line to penultimate in order to just gather relevant lines

    with open(info_filename, "r") as f:
        info = f.readlines()[2:-2]

    # remove whitespaces and newlines
    info = [x.split() for x in info]

    column_names = []
    for line in info:
        if len(line) < 4:
            continue

        column_names.append(line[1])
        column_names.append(line[-4])

    return column_names

def read_en_info(energy_filenames, info_filename):

    data = read_en(energy_filenames)
    column_names = read_info(info_filename)

    # combine data and column_names
    data = pd.concat(data, axis=1)
    data.columns = column_names

    return data, column_names