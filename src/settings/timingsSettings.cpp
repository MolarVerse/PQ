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
void TimingsSettings::setTimeStepByUser(const Time& timeStep)
{
    _timeStep.setByUser(timeStep);
}

/**
 * @brief Set the step count
 *
 * @param stepCount
 */
void TimingsSettings::setStartingStepCountByUser(size_t stepCount)
{
    _startingStepCount.setByUser(stepCount);
}

/**
 * @brief Set the number of steps
 *
 * @param numberOfSteps
 */
void TimingsSettings::setNumberOfStepsByUser(size_t numberOfSteps)
{
    _numberOfSteps.setByUser(numberOfSteps);
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the time step
 *
 * @return const Time&
 */
const Time& TimingsSettings::getTimeStep() const { return _timeStep.get(); }

/**
 * @brief get the step count
 *
 * @return size_t
 */
size_t TimingsSettings::getStartingStepCount() const
{
    return _startingStepCount.get();
}

/**
 * @brief get the number of steps
 *
 * @return size_t
 */
size_t TimingsSettings::getNumberOfSteps() const
{
    return _numberOfSteps.get();
}
