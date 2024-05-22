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

#include "timingsOutput.hpp"

#include <format>   // for std::format

#include "globalTimer.hpp"   // for timings::GlobalTimer

using namespace output;

/**
 * @brief Write the timings to the output file
 *
 * @param timer The timer object
 */
void TimingsOutput::write(timings::GlobalTimer &timer)
{
    timer.sortTimers();

    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\n",
        "Section",
        "Time [s]",
        "Time [%]"
    );

    // write a line consisting only of '-'
    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\n",
        std::string(30, '-'),
        std::string(10, '-'),
        std::string(10, '-')
    );

    _fp << "\n";

    // write the simulation timer
    _fp << std::format(
        "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
        "Total",
        timer.calculateElapsedTime(),
        100.0
    );

    _fp << "\n";

    // write the execution timers
    for (const auto &section : timer.getTimers())
    {
        const auto name       = section.getTimerName();
        const auto time       = section.calculateElapsedTime();
        const auto percentage = (time / timer.calculateElapsedTime()) * 100.0;

        _fp << std::format(
            "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
            name,
            time,
            percentage
        );
    }
}