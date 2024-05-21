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

/**
 * @brief end the timer
 *
 */
void TimingsSection::endTimer()
{
    _end           = std::chrono::high_resolution_clock::now();
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
    return double(std::chrono::duration_cast<ms>(_totalTime).count());
}

double TimingsSection::calculateAverageLoopTime() const
{
    auto time = double(std::chrono::duration_cast<ns>(_totalTime).count());
    time      = time * 1e-9 / double(_steps);

    return time;
}

/**
 * @brief calculates the loop time in s
 *
 */
double TimingsSection::calculateLoopTime() const
{
    auto time = double(std::chrono::duration_cast<ns>(_lastStepTime).count());
    time      = time * 1e-9;

    return time;
}