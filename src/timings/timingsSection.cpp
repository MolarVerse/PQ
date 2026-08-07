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

#include "timingsSection.hpp"

#include <chrono>   // IWYU pragma: keep for time_point, milliseconds, nanoseconds

using namespace timings;
using namespace std::chrono;

using ms = milliseconds;
using ns = nanoseconds;

/**
 * @brief Construct a new Timings Section:: Timings Section object
 *
 * @param name
 */
TimingsSection::TimingsSection(const std::string_view name) : _name(name) {}

/**
 * @brief
 *
 */
void TimingsSection::beginTimer() { _start = high_resolution_clock::now(); }

/**
 * @brief end the timer
 *
 */
void TimingsSection::endTimer()
{
    _end           = high_resolution_clock::now();
    _steps         = _steps + 1;
    _totalTime    += _end - _start;
    _lastStepTime  = _end - _start;
}

/**
 * @brief calculates the elapsed time in ms
 *
 */
double TimingsSection::calculateElapsedTime() const
{
    return double(duration_cast<ns>(_totalTime).count()) * 1.0e-6;
}

double TimingsSection::calculateAverageLoopTime() const
{
    auto time = double(duration_cast<ns>(_totalTime).count());
    time      = time * 1.0e-9 / double(_steps);

    return time;
}

/**
 * @brief calculates the loop time in s
 *
 */
double TimingsSection::calculateLoopTime() const
{
    auto time = double(duration_cast<ns>(_lastStepTime).count());
    time      = time * 1e-9;

    return time;
}

/**
 * @brief get the name of the timings section
 *
 * @return std::string
 */
std::string TimingsSection::getName() const { return _name; }
