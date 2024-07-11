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

#include "nonCoulombInputParser.hpp"

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "stringUtilities.hpp"     // for toLowerCopy

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser
 * Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) noncoulomb <string>
 *
 * @param engine
 */
NonCoulombInputParser::NonCoulombInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("noncoulomb"),
        bind_front(&NonCoulombInputParser::parseNonCoulombType, this),
        false
    );
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
 * @throws InputFileException if invalid nonCoulomb type
 */
void NonCoulombInputParser::parseNonCoulombType(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto type = toLowerCopy(lineElements[2]);

    using enum NonCoulombType;

    if (type == "guff")
        PotentialSettings::setNonCoulombType(GUFF);

    else if (type == "lj")
        PotentialSettings::setNonCoulombType(LJ);

    else if (type == "buck")
        PotentialSettings::setNonCoulombType(BUCKINGHAM);

    else if (type == "morse")
        PotentialSettings::setNonCoulombType(MORSE);

    else
        throw InputFileException(format(
            "Invalid nonCoulomb type \"{}\" at line {} in input file.\n"
            "Possible options are: lj, buck, morse and guff",
            lineElements[2],
            lineNumber
        ));
}