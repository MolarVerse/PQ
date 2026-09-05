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

#include "constants/conversionFactors.hpp"
#include "globalTimer.hpp"   // for GlobalTimer

using namespace out;
using namespace timings;

/**
 * @brief Write the timings to the output file
 *
 * @param timer The timer object
 */
void TimingsOutput::write()
{
    const auto timers = timings::GlobalTimer::get().sortTimers();

    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\n",
        "Section",
        "Time [s]",
        "Time [%]"
    );

    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
    // write a line consisting only of '-'
    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\n",
        std::string(30, '-'),
        std::string(10, '-'),
        std::string(10, '-')
    );

    _fp << "\n";

    const auto elapsedTime = timings::GlobalTimer::get().calculateElapsedTime();

    // write the simulation timer
    _fp << std::format(
        "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
        "Total",
        elapsedTime * constants::MS_TO_S,
        100.0
    );
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

    _fp << "\n";

    // write the execution timers
    for (const auto &section : timers)
    {
        const auto name       = section.getTimerName();
        const auto time       = section.calculateElapsedTime();
        const auto percentage = (time / elapsedTime) * 100.0;

        _fp << std::format(
            "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
            name,
            time * constants::MS_TO_S,
            percentage
        );
    }

    _fp << "\n";
    _fp << "\n";
    _fp << "\n";
    _fp << "\n";

    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\t{:>10}\n",
        "Section",
        "Time [s]",
        "Time [%]",
        "RelT [%]"
    );

    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
    // write a line consisting only of '-'
    _fp << std::format(
        "{:<30}\t{:>10}\t{:>10}\t{:>10}\n",
        std::string(30, '-'),
        std::string(10, '-'),
        std::string(10, '-'),
        std::string(10, '-')
    );

    _fp << "\n";

    // write the simulation timer
    _fp << std::format(
        "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
        "Total",
        elapsedTime * constants::MS_TO_S,
        100.0,
        100.0
    );
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

    _fp << "\n";

    // write the execution timers
    for (const auto &section : timers)
    {
        auto subsections = section.getTimingDetails();

        if (subsections.empty())
            continue;

        const auto name       = section.getTimerName();
        const auto time       = section.calculateElapsedTime();
        const auto percentage = (time / elapsedTime) * 100.0;

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
        _fp << std::format(
            "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
            name,
            time * constants::MS_TO_S,
            percentage,
            100.0
        );
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

        for (const auto &subSection : subsections)
        {
            const auto subName          = subSection.getName();
            const auto subTime          = subSection.calculateElapsedTime();
            const auto subPercentage    = (subTime / time) * 100.0;
            const auto subTotPercentage = (subTime / elapsedTime) * 100.0;

            _fp << std::format(
                "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
                subName,
                subTime * constants::MS_TO_S,
                subTotPercentage,
                subPercentage
            );
        }

        _fp << "\n";
    }

    _fp << std::flush;
}
