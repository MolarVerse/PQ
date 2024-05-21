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

#include "timings.hpp"

#include "timingsSettings.hpp"

using namespace timings;

/*
 * @brief constructor
 */
Timings::Timings()
{
    _timingDetails.emplace_back(TimingsManager("Writing"));
    _timingDetails.emplace_back(TimingsManager("Setup"));
}

/**
 * @brief calculates the total simulation time in fs
 *
 */
double Timings::calculateTotalSimTime(const size_t step) const
{
    const auto totalSteps = step + _stepCount;

    return totalSteps * settings::TimingsSettings::getTimeStep();
}

/**
 * @brief calculates the elapsed time in ms
 *
 */
long Timings::calculateElapsedTime() const
{
    return duration_cast<ms>(_end - _start).count();
}

/**
 * @brief calculates the loop time in s
 *
 */
double Timings::calculateLoopTime(const size_t numberOfSteps)
{
    _end          = high_resolution_clock::now();
    auto loopTime = double(duration_cast<ns>(_end - _start).count());
    loopTime      = loopTime * 1e-9 / double(numberOfSteps);

    return loopTime;
}