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

#include "timingsSettings.hpp"

using settings::TimingsSettings;

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief Set the time step
 *
 * @param timeStep
 */
void TimingsSettings::setTimeStep(const double timeStep)
{
    _timeStep = timeStep;
}

/**
 * @brief Set the step count
 *
 * @param stepCount
 */
void TimingsSettings::setStepCount(const size_t stepCount)
{
    _stepCount = stepCount;
}

/**
 * @brief Set the number of steps
 *
 * @param numberOfSteps
 */
void TimingsSettings::setNumberOfSteps(const size_t numberOfSteps)
{
    _numberOfSteps = numberOfSteps;
}

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief get the time step
 *
 * @return double
 */
double TimingsSettings::getTimeStep() { return _timeStep; }

/**
 * @brief get the step count
 *
 * @return size_t
 */
size_t TimingsSettings::getStepCount() { return _stepCount; }

/**
 * @brief get the number of steps
 *
 * @return size_t
 */
size_t TimingsSettings::getNumberOfSteps() { return _numberOfSteps; }