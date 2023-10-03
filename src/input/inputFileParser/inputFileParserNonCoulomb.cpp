/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "inputFileParserNonCoulomb.hpp"

#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "stringUtilities.hpp"     // for toLowerCopy

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace input;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) noncoulomb <string>
 *
 * @param engine
 */
InputFileParserNonCoulomb::InputFileParserNonCoulomb(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("noncoulomb"), bind_front(&InputFileParserNonCoulomb::parseNonCoulombType, this), false);
}

/**
 * @brief Parse the nonCoulombic type of the guff.dat file
 *
 * @details Possible options are:
 * 1) "guff"  - guff.dat file is used (default)
 * 2) "lj"
 * 3) "buck"
 * 4) "morse"
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if invalid nonCoulomb type
 */
void InputFileParserNonCoulomb::parseNonCoulombType(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto type = utilities::toLowerCopy(lineElements[2]);

    if (type == "guff")
        settings::PotentialSettings::setNonCoulombType("guff");
    else if (type == "lj")
        settings::PotentialSettings::setNonCoulombType("lj");
    else if (type == "buck")
        settings::PotentialSettings::setNonCoulombType("buck");
    else if (type == "morse")
        settings::PotentialSettings::setNonCoulombType("morse");
    else
        throw customException::InputFileException(
            format("Invalid nonCoulomb type \"{}\" at line {} in input file. Possible options are: lj, buck, morse and guff",
                   lineElements[2],
                   lineNumber));
}