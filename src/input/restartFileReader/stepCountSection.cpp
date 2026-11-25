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
#include <stdexcept>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "settings.hpp"
#include "stringUtilities.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace input::restartFile
{

    /**
     * @brief processes the step count section of the rst file
     *
     * @details The step count section is a header section and must have 2
     * or 4 elements:
     * 1. keyword "step"
     * 2. step count
     * 3. (optional) keyword "simulation_time"
     * 4. (optional) total simulation time of all previous steps in ps
     *
     * @param lineElements all elements of the line
     * @param lineNumber the line number in the rst file being processed
     * @param settings reference to the Settings object
     * @param simulationBox reference to the SimulationBox object
     *
     * @throws RstFileException if the number of elements in the line is not 2
     * or 4
     * @throws RstFileException if the step count is negative
     * @throws RstFileException if the third element is not "simulation_time"
     * @throws RstFileException if the simulation time is not a non-negative
     * number
     */
    void StepCountSection::process(
        std::vector<std::string> &lineElements,
        size_t                   &lineNumber,
        Settings                 &settings,
        molsys::SimulationBox & /*simulationBox*/
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

        settings.timings.setStartingStepCountByUser(
            static_cast<size_t>(stepCount)
        );

        if (lineElements.size() == 4)
        {
            const auto keyword =
                utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

            if (keyword != "simulation_time")
            {
                throw exc::RstFileException(
                    std::format(
                        "Error in line {}: Expected keyword "
                        "\"simulation_time\" after the step count but got "
                        "\"{}\"",
                        lineNumber,
                        lineElements[2]
                    )
                );
            }

            double startingTime = 0.0;

            try
            {
                std::size_t pos = 0;
                startingTime    = std::stod(lineElements[3], &pos);

                if (pos != lineElements[3].size())
                {
                    throw std::invalid_argument(lineElements[3]);
                }
            }
            catch (const std::logic_error &)
            {
                throw exc::RstFileException(
                    std::format(
                        "Error in line {}: Invalid simulation time \"{}\"",
                        lineNumber,
                        lineElements[3]
                    )
                );
            }

            if (!(startingTime >= 0.0))
            {
                throw exc::RstFileException(
                    std::format(
                        "Error in line {}: Simulation time must not be "
                        "negative",
                        lineNumber
                    )
                );
            }

            settings.timings.setStartingTimeByUser(Time(startingTime * ps));
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
