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

#include <format>

#include "constants/conversionFactors.hpp"
#include "globalTimer.hpp"

namespace out
{

    /**
     * @brief Write the timings to the output file
     *
     */
    void TimingsOutput::write()
    {
        const auto timers = timings::GlobalTimer::get().sortTimers();

        _getFilePointer() << std::format(
            "{:<30}\t{:>10}\t{:>10}\n",
            "Section",
            "Time [s]",
            "Time [%]"
        );

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,
        // readability-magic-numbers) write a line consisting only of '-'
        _getFilePointer() << std::format(
            "{:<30}\t{:>10}\t{:>10}\n",
            std::string(30, '-'),
            std::string(10, '-'),
            std::string(10, '-')
        );

        _getFilePointer() << "\n";

        const auto elapsedTime =
            timings::GlobalTimer::get().calculateElapsedTime();

        // write the simulation timer
        _getFilePointer() << std::format(
            "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
            "Total",
            elapsedTime * MS_TO_S,
            100.0
        );
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,
        // readability-magic-numbers)

        _getFilePointer() << "\n";

        // write the execution timers
        for (const auto &section : timers)
        {
            const auto name       = section.getTimerName();
            const auto time       = section.calculateElapsedTime();
            const auto percentage = (time / elapsedTime) * 100.0;

            _getFilePointer() << std::format(
                "{:<30}\t{:>10.3f}\t{:>10.3f}\n",
                name,
                time * MS_TO_S,
                percentage
            );
        }

        _getFilePointer() << "\n";
        _getFilePointer() << "\n";
        _getFilePointer() << "\n";
        _getFilePointer() << "\n";

        _getFilePointer() << std::format(
            "{:<30}\t{:>10}\t{:>10}\t{:>10}\n",
            "Section",
            "Time [s]",
            "Time [%]",
            "RelT [%]"
        );

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,
        // readability-magic-numbers) write a line consisting only of '-'
        _getFilePointer() << std::format(
            "{:<30}\t{:>10}\t{:>10}\t{:>10}\n",
            std::string(30, '-'),
            std::string(10, '-'),
            std::string(10, '-'),
            std::string(10, '-')
        );

        _getFilePointer() << "\n";

        // write the simulation timer
        _getFilePointer() << std::format(
            "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
            "Total",
            elapsedTime * MS_TO_S,
            100.0,
            100.0
        );
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,
        // readability-magic-numbers)

        _getFilePointer() << "\n";

        // write the execution timers
        for (const auto &section : timers)
        {
            auto subsections = section.getTimingDetails();

            if (subsections.empty())
                continue;

            const auto name       = section.getTimerName();
            const auto time       = section.calculateElapsedTime();
            const auto percentage = (time / elapsedTime) * 100.0;

            // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,
            // readability-magic-numbers)
            _getFilePointer() << std::format(
                "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
                name,
                time * MS_TO_S,
                percentage,
                100.0
            );
            // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,
            // readability-magic-numbers)

            for (const auto &subSection : subsections)
            {
                const auto subName          = subSection.getName();
                const auto subTime          = subSection.calculateElapsedTime();
                const auto subPercentage    = (subTime / time) * 100.0;
                const auto subTotPercentage = (subTime / elapsedTime) * 100.0;

                _getFilePointer() << std::format(
                    "{:<30}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n",
                    subName,
                    subTime * MS_TO_S,
                    subTotPercentage,
                    subPercentage
                );
            }

            _getFilePointer() << "\n";
        }

        _getFilePointer() << std::flush;
    }

}   // namespace out
