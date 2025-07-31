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

#include <cstddef>   // for size_t
#include <format>    // for format
#include <string>    // for string, stoi
#include <vector>    // for vector

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for RstFileException
#include "timingsSettings.hpp"   // for TimingsSettings

using namespace input::restartFile;
using namespace engine;
using namespace settings;
using namespace customException;

/**
 * @brief processes the step count section of the rst file
 *
 * @details The step count section is a header section and must have 2 elements:
 * 1. keyword "step"
 * 2. step count
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the line is not 2
 * @throws RstFileException if the step count is negative
 */
void StepCountSection::process(std::vector<std::string> &lineElements, Engine &)
{
    if (lineElements.size() != 2)
        throw RstFileException(
            std::format(
                "Error in line {}: Step count section must have 2 elements",
                _lineNumber
            )
        );

    auto stepCount = stoi(lineElements[1]);

    if (stepCount < 0)
        throw RstFileException(
            std::format(
                "Error in line {}: Step count must be positive",
                _lineNumber
            )
        );

    TimingsSettings::setStepCount(size_t(stepCount));
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