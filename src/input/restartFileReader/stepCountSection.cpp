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

#include "stepCountSection.hpp"

#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include "engine.hpp"
#include "exceptions.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace input::restartFile
{

    /**
     * @brief constructs a StepCountSection with the given TimingsSettings
     *
     * @param timingsSettings reference to the TimingsSettings object
     */
    StepCountSection::StepCountSection(TimingsSettings &timingsSettings)
        : _timingsSettings(timingsSettings)
    {
    }

    /**
     * @brief processes the step count section of the rst file
     *
     * @details The step count section is a header section and must have 2
     * elements:
     * 1. keyword "step"
     * 2. step count
     *
     * @param lineElements all elements of the line
     * @param engine object containing the engine
     * @param lineNumber the line number in the rst file being processed
     *
     * @throws RstFileException if the number of elements in the line is not 2
     * @throws RstFileException if the step count is negative
     */
    void StepCountSection::process(
        std::vector<std::string> &lineElements,
        engine::Engine & /*engine*/,
        size_t &lineNumber
    )
    {
        if (lineElements.size() != 2 && lineElements.size() != 4)
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {}: Step count section must have 2 or 4 "
                    "elements",
                    lineNumber
                )
            );
        }

        const auto stepCount = stoi(lineElements[1]);

        if (stepCount < 0)
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {}: Step count must be positive",
                    lineNumber
                )
            );
        }

        _timingsSettings->setStartingStepCountByUser(
            static_cast<size_t>(stepCount)
        );

        if (lineElements.size() == 4)
        {
            const auto startingTime = stod(lineElements[3]) * ps;
            _timingsSettings->setStartingTimeByUser(Time(startingTime));
        }
    }

    /**
     * @brief returns the keyword of the section
     *
     * @return "step"
     */
    std::string StepCountSection::keyword() { return "step"; }

    /**
     * @brief returns if the section is a header
     *
     * @return true
     */
    bool StepCountSection::isHeader() { return true; }

}   // namespace input::restartFile
