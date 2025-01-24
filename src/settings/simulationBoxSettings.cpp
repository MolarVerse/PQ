/*****************************************************************************
<GPL_HEADER>

    PQ
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
******************************************************************************/

#include "simulationBoxSettings.hpp"

using settings::SimulationBoxSettings;

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief Set the density set
 *
 * @param densitySet
 */
void SimulationBoxSettings::setDensitySet(const bool densitySet)
{
    _isDensitySet = densitySet;
}

/**
 * @brief Set the box set
 *
 * @param boxSet
 */
void SimulationBoxSettings::setBoxSet(const bool boxSet) { _isBoxSet = boxSet; }

/**
 * @brief Set the initialize velocities
 *
 * @param initVelocities
 */
void SimulationBoxSettings::setInitializeVelocities(const bool initVelocities)
{
    _initializeVelocities = initVelocities;
}

/**
 * @brief Set zeroVelocities to indicate all zero entries in .rst file for
 * velocities
 *
 * @param zeroVelocities
 */
void SimulationBoxSettings::setZeroVelocities(const bool zeroVelocities)
{
    _zeroVelocities = zeroVelocities;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get if the density is set
 *
 * @return true
 * @return false
 */
bool SimulationBoxSettings::getDensitySet() { return _isDensitySet; }

/**
 * @brief get if the box is set
 *
 * @return true
 * @return false
 */
bool SimulationBoxSettings::getBoxSet() { return _isBoxSet; }

/**
 * @brief get if the velocities are initialized
 *
 * @return true
 * @return false
 */
bool SimulationBoxSettings::getInitializeVelocities()
{
    return _initializeVelocities;
}

/**
 * @brief get if velocities in .rst file are all zero entries
 *
 * @return true
 * @return false
 */
bool SimulationBoxSettings::getZeroVelocities() { return _zeroVelocities; }