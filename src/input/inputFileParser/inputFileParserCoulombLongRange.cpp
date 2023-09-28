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

#include "inputFileParserCoulombLongRange.hpp"

#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "stringUtilities.hpp"     // for toLowerCopy

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace input;

/**
 * @brief Construct a new Input File Parser Coulomb Long Range:: Input File Parser Coulomb Long Range object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) long_range <string>
 * 2) wolf_param <double>
 *
 * @param engine
 */
InputFileParserCoulombLongRange::InputFileParserCoulombLongRange(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("long_range"), bind_front(&InputFileParserCoulombLongRange::parseCoulombLongRange, this), false);
    addKeyword(std::string("wolf_param"), bind_front(&InputFileParserCoulombLongRange::parseWolfParameter, this), false);
}

/**
 * @brief Parse the coulombic long-range correction used in the simulation
 *
 * @details Possible options are:
 * 1) "none" - no long-range correction is used (default) = shifted potential
 * 2) "wolf" - wolf long-range correction is used
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if coulombic long-range correction is not valid - currently only none and wolf are
 * supported
 */
void InputFileParserCoulombLongRange::parseCoulombLongRange(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto type = utilities::toLowerCopy(lineElements[2]);

    if (type == "none")
        settings::PotentialSettings::setCoulombLongRangeType("none");

    else if (type == "wolf")
        settings::PotentialSettings::setCoulombLongRangeType("wolf");

    else
        throw customException::InputFileException(format(
            R"(Invalid long-range type for coulomb correction "{}" at line {} in input file - possible options are "none", "wolf")",
            lineElements[2],
            lineNumber));
}

/**
 * @brief parse the wolf parameter used in the simulation
 *
 * @details default value is 0.25
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if wolf parameter is negative
 */
void InputFileParserCoulombLongRange::parseWolfParameter(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto wolfParameter = stod(lineElements[2]);

    if (wolfParameter < 0.0)
        throw customException::InputFileException("Wolf parameter cannot be negative");

    settings::PotentialSettings::setWolfParameter(wolfParameter);
}