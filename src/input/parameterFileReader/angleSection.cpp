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

#include "angleSection.hpp"

#include <format>   // for format

#include "angleType.hpp"                     // for AngleType
#include "constants/conversionFactors.hpp"   // for _DEG_TO_RAD_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for ParameterFileException
#include "forceFieldClass.hpp"               // for ForceField

using namespace input::parameterFile;
using namespace engine;
using namespace customException;
using namespace forceField;
using namespace constants;

/**
 * @brief returns the keyword of the angle section
 *
 * @return "angles"
 */
std::string AngleSection::keyword() { return "angles"; }

/**
 * @brief processes one line of the angle section of the parameter file and adds
 * the angle type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. angleTypeId
 * 2. equilibriumAngle
 * 3. forceConstant
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 3
 */
void AngleSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 3)
        throw ParameterFileException(
            std::format(
                "Wrong number of arguments in parameter file angle section at "
                "line "
                "{} - number of elements has to be 3!",
                _lineNumber
            )
        );

    auto id               = stoul(lineElements[0]);
    auto equilibriumAngle = stod(lineElements[1]) * _DEG_TO_RAD_;
    auto forceConstant    = stod(lineElements[2]);

    auto angleType = AngleType(id, equilibriumAngle, forceConstant);

    engine.getForceField().addAngleType(angleType);
}