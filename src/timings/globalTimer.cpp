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
#include <vector>

#include "timer.hpp"

using namespace timings;

/**
 * @brief Construct a new Global Timer:: Global Timer object
 *
 */
GlobalTimer::GlobalTimer()
{
    for (const auto id : TimerIdMeta::values_view())
        _timers.at(static_cast<size_t>(id)) = Timer(id);

    _getSimulationTimer().startTimingsSection();
}

/**
 * @brief Get the Simulation Timer object
 *
 * @return Timer&
 */
Timer& GlobalTimer::_getSimulationTimer()
{
    return _timers.at(static_cast<size_t>(TimerId::Simulation));
}

/**
 * @brief Get the Simulation Timer object (const version)
 *
 * @return const Timer&
 */
const Timer& GlobalTimer::_getSimulationTimer() const
{
    return _timers.at(static_cast<size_t>(TimerId::Simulation));
}

/**
 * @brief Get the Timer object for a specific TimerId
 *
 * @param id
 * @return Timer&
 */
Timer& GlobalTimer::_getTimer(const TimerId id)
{
    return _timers.at(static_cast<size_t>(id));
}

/**
 * @brief Get the Timer object for a specific TimerId (const version)
 *
 * @param id
 * @return const Timer&
 */
const Timer& GlobalTimer::_getTimer(const TimerId id) const
{
    return _timers.at(static_cast<size_t>(id));
}

/**
 * @brief calculates the loop time of the simulation
 *
 * @return double
 */
double GlobalTimer::calculateLoopTime() const
{
    return _getSimulationTimer().calculateLoopTime();
}

/**
 * @brief calculates the elapsed time of the simulation
 *
 * @return double
 */
double GlobalTimer::calculateElapsedTime() const
{
    return _getSimulationTimer().calculateElapsedTime();
}

/**
 * @brief sorts the timers
 *
 * @return std::vector<Timer>
 *
 */
std::vector<Timer> GlobalTimer::sortTimers() const
{
    std::vector<Timer> sortedTimers(_timers.begin(), _timers.end());

    for (auto timer : sortedTimers) timer.sortTimingsSections();

    std::ranges::sort(
        sortedTimers,
        [](const Timer& a, const Timer& b)
        { return a.calculateElapsedTime() > b.calculateElapsedTime(); }
    );

    return sortedTimers;
}

/**
 * @brief stop the simulation timer
 *
 */
void GlobalTimer::stopSimulationTimer()
{
    _getSimulationTimer().stopTimingsSection();
}

/**
 * @brief stop and restart the simulation timer
 *
 */
void GlobalTimer::stopAndRestartSimulationTimer()
{
    _getSimulationTimer().stopTimingsSection();
    _getSimulationTimer().startTimingsSection();
}

/**
 * @brief get a scoped timer for a specific timer id and section name
 *
 * @param id
 * @param sectionName
 * @return TimingsSectionGuard
 */
TimingsSectionGuard GlobalTimer::scoped(
    TimerId            id,
    const std::string& sectionName
)
{
    return _getTimer(id).scoped(sectionName);
}
