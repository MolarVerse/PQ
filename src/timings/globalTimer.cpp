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

#include "globalTimer.hpp"

#include <algorithm>   // for ranges::sort
#include <ranges>      // for ranges::sort

#include "timer.hpp"

using namespace timings;

/**
 * @brief adds a simulation timer
 *
 * @param simulationTimer
 */
void GlobalTimer::addSimulationTimer(const Timer& simulationTimer)
{
    _simulationTimer = simulationTimer;
}

/**
 * @brief calculates the loop time of the simulation
 *
 * @return double
 */
double GlobalTimer::calculateLoopTime() const
{
    return _simulationTimer.calculateLoopTime();
}

/**
 * @brief calculates the elapsed time of the simulation
 *
 * @return double
 */
double GlobalTimer::calculateElapsedTime() const
{
    return _simulationTimer.calculateElapsedTime();
}

/**
 * @brief sorts the timers
 *
 */
void GlobalTimer::sortTimers()
{
    for (auto& timer : _timers) timer.sortTimingsSections();

    std::ranges::sort(
        _timers,
        [](const Timer& a, const Timer& b)
        { return a.calculateElapsedTime() > b.calculateElapsedTime(); }
    );
}

/**
 * @brief get the timers
 *
 * @return const std::vector<Timer>&
 */
const std::vector<Timer>& GlobalTimer::getTimers() const { return _timers; }