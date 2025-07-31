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

#include "typesSection.hpp"

#include <format>   // for format

#include "exceptions.hpp"          // for ParameterFileException
#include "potentialSettings.hpp"   // for PotentialSettings

using namespace input::parameterFile;
using namespace engine;
using namespace customException;
using namespace settings;

/**
 * @brief returns the keyword of the section
 *
 * @return std::string
 */
std::string TypesSection::keyword() { return "types"; }

/**
 * @brief Overwrites process function of ParameterFileSection base class. It
 * just forwards the call to processSection.
 *
 * @param lineElements
 * @param engine
 */
void TypesSection::process(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    processSection(lineElements, engine);
}

/**
 * @brief process types section and sets the scale factors for the 1-4
 * interactions in potentialSettings
 *
 * @details The types section is used to set the scale factors for the 1-4
 * interactions. The line must have the following form for backward
 * compatibility: 1) dummy 2) dummy 3) dummy 4) dummy 5) dummy 6) dummy 7)
 * scaleCoulomb 8) scaleVanDerWaals
 *
 * @param lineElements
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 8
 * @throw ParameterFileException if scaleCoulomb is not between
 * 0 and 1
 * @throw ParameterFileException if scaleVanDerWaals is not
 * between 0 and 1
 */
void TypesSection::processSection(pq::strings &lineElements, Engine &)
{
    if (lineElements.size() != 8)
        throw ParameterFileException(
            std::format(
                "Wrong number of arguments in parameter file types section at "
                "line "
                "{} - number of elements has to be 8!",
                _lineNumber
            )
        );

    const auto scaleCoulomb     = stod(lineElements[6]);
    const auto scaleVanDerWaals = stod(lineElements[7]);

    if (scaleCoulomb < 0.0 || scaleCoulomb > 1.0)
        throw ParameterFileException(
            std::format(
                "Wrong scaleCoulomb in parameter file types section at line {} "
                "- "
                "has to be between 0 and 1!",
                _lineNumber
            )
        );

    if (scaleVanDerWaals < 0.0 || scaleVanDerWaals > 1.0)
        throw ParameterFileException(
            std::format(
                "Wrong scaleVanDerWaals in parameter file types section at "
                "line {} "
                "- has to be between 0 and 1!",
                _lineNumber
            )
        );

    PotentialSettings::setScale14Coulomb(scaleCoulomb);
    PotentialSettings::setScale14VanDerWaals(scaleVanDerWaals);
}