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

import numpy as np

# Function that selects called feature and returns tuple of datapoints and name of feature
def get_feature(i, value, data, kernelSize):
    if i == 1 and value == 1:
        return (runningAverage(data, kernelSize), "Running Average" + " (" + str(kernelSize) + ")")
    elif i == 2 and value == 1:
        return (overallAverage(data), "Overall Average")
    elif i == 3 and value == 1:
        return (integrationAverage(data), "Integratation Average")
    else:
        # raise Exception("Feature not yet implemented")
        return None


# Features:
#    -running average
#    -overall average
#    -integralation average over the number of datapoints
def runningAverage(data, kernelSize):
    return data.rolling(kernelSize).mean()


def overallAverage(data):
    average = data.mean()
    return np.repeat(average, len(data))

def integrationAverage(data):
    list = []
    sum = 0
    for (i, y) in enumerate(data):
        sum += y
        list.append(sum / (i+1))
    return np.array(list)
