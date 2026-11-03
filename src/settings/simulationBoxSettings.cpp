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

/**
 * @brief Construct a new Simulation Box Settings object
 *
 */
SimulationBoxSettings::SimulationBoxSettings()
    : _initializeVelocities(InitVelocities::FALSE)
{
}

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/
/**
 * @brief Set the density
 *
 * @param density
 */
void SimulationBoxSettings::setDensity(double density)
{
    _density.set(density);
}

/**
 * @brief Set the initialize velocities
 *
 * @param initializeVelocities
 */
void SimulationBoxSettings::setInitializeVelocities(
    InitVelocities initializeVelocities
)
{
    _initializeVelocities.set(initializeVelocities);
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/
/**
 * @brief Get the density
 *
 * @return double
 */
double SimulationBoxSettings::getDensity() const { return _density.get(); }

/**
 * @brief get if the density is set
 *
 * @return true
 * @return false
 */
bool SimulationBoxSettings::isDensitySet() const { return _density.isSet(); }

/**
 * @brief get if the velocities are initialized
 *
 * @return TRUE
 * @return FALSE
 * @return FORCE
 */
InitVelocities SimulationBoxSettings::getInitializeVelocities() const
{
    return _initializeVelocities.get();
}
