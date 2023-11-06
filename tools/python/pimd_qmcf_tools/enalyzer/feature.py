"""
*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper,  Josef M. Gallmetzer

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

import numpy as np

# Function that selects called feature and returns tuple of datapoints and name of feature
def get_feature(i, value, data, kernel_size):
    if i == 1 and value == 1:
        return (running_average(data, kernel_size), "Running Average" + " (" + str(kernel_size) + ")")
    elif i == 2 and value == 1:
        return (overall_average(data), "Overall Average")
    elif i == 3 and value == 1:
        return (integration_average(data), "Integratation Average")
    else:
        return None


# Features:
#    -running average
#    -overall average
#    -integration average over the number of datapoints
def running_average(data, kernel_size):
    return data.rolling(kernel_size).mean()


def overall_average(data):
    average = data.mean()
    return np.repeat(average, len(data))

def integration_average(data):
    _list = []
    _sum = 0
    for (i, y) in enumerate(data):
        _sum += y
        _list.append(_sum / (i+1))
    return np.array(_list)
