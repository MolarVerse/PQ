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

#include "exceptions.hpp"
#include "timingsSettings.hpp"

using namespace timings;

/*
 * @brief constructor
 */
Timings::Timings(const std::string_view name) : _name(name) {}

/**
 * @brief calculates the elapsed time in ms
 *
 */
double Timings::calculateElapsedTime() const
{
    auto elapsedTime = 0;

    for (const auto& timing : _timingDetails)
        elapsedTime += timing.calculateElapsedTime();

    return elapsedTime;
}

/**
 * @brief calculates the loop time in s
 *
 */
double Timings::calculateLoopTime()
{
    auto loopTime = 0.0;

    for (const auto& timing : _timingDetails)
        loopTime += timing.calculateLoopTime();

    return loopTime;
}

/**
 * @brief get TimingsManager by name
 *
 */
TimingsManager Timings::getTimingsManager(const std::string_view name) const
{
    const auto index = findTimeManagerIndex(name);

    if (index == _timingDetails.size())
        throw customException::CustomException("Timer not found");

    return _timingDetails[index];
}

/**
 * @brief starts a new timer
 *
 */
void Timings::startTimeManager(const std::string_view name)
{
    const auto index = findTimeManagerIndex(name);

    if (index == _timingDetails.size())
    {
        _timingDetails.emplace_back(name);
        _timingDetails.back().beginTimer();
    }
    else
        _timingDetails[index].beginTimer();
}

/**
 * @brief find timeManager by name
 *
 */
size_t Timings::findTimeManagerIndex(const std::string_view name) const
{
    for (size_t i = 0; i < _timingDetails.size(); ++i)
        if (_timingDetails[i].getName() == name)
            return i;

    return _timingDetails.size();
}

/**
 * @brief stops a timer
 *
 */
void Timings::stopTimeManager(const std::string_view name)
{
    const auto index = findTimeManagerIndex(name);

    if (index == _timingDetails.size())
        throw customException::CustomException("Timer not found");

    _timingDetails[index].endTimer();
}
