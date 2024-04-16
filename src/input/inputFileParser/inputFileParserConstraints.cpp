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

#include "inputFileParserConstraints.hpp"

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "constraints.hpp"          // for Constraints
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for InputFileException

#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace input;

/**
 * @brief Construct a new Input File Parser Constraints:: Input File Parser Constraints object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) shake <on/off>
 * 2) shake-tolerance <double>
 * 3) shake-iter <size_t>
 * 4) rattle-iter <size_t>
 * 5) rattle-tolerance <double>
 *
 * @param engine
 */
InputFileParserConstraints::InputFileParserConstraints(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("shake"), bind_front(&InputFileParserConstraints::parseShakeActivated, this), false);
    addKeyword(std::string("shake-tolerance"), bind_front(&InputFileParserConstraints::parseShakeTolerance, this), false);
    addKeyword(std::string("shake-iter"), bind_front(&InputFileParserConstraints::parseShakeIteration, this), false);
    addKeyword(std::string("rattle-iter"), bind_front(&InputFileParserConstraints::parseRattleIteration, this), false);
    addKeyword(std::string("rattle-tolerance"), bind_front(&InputFileParserConstraints::parseRattleTolerance, this), false);
}

/**
 * @brief parsing if shake is activated
 *
 * @details Possible options are:
 * 1) "on"  - shake is activated
 * 2) "off" - shake is deactivated (default)
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if keyword is not valid - currently only on and off are supported
 */
void InputFileParserConstraints::parseShakeActivated(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
        _engine.getConstraints().activate();
    else if (lineElements[2] == "off")
        _engine.getConstraints().deactivate();
    else
    {
        auto message = format(R"(Invalid shake keyword "{}" at line {} in input file\n Possible keywords are "on" and "off")",
                              lineElements[2],
                              lineNumber);
        throw customException::InputFileException(message);
    }
}

/**
 * @brief parsing shake tolerance
 *
 * @details default value is 1e-8
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if tolerance is negative
 */
void InputFileParserConstraints::parseShakeTolerance(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw customException::InputFileException("Shake tolerance must be positive");

    settings::ConstraintSettings::setShakeTolerance(tolerance);
}

/**
 * @brief parsing shake iteration
 *
 * @details default value is 20
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if iteration is negative
 */
void InputFileParserConstraints::parseShakeIteration(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw customException::InputFileException("Maximum shake iterations must be positive");

    settings::ConstraintSettings::setShakeMaxIter(size_t(iteration));
}

/**
 * @brief parsing rattle tolerance
 *
 * @details default value is 1e-8
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if tolerance is negative
 */
void InputFileParserConstraints::parseRattleTolerance(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw customException::InputFileException("Rattle tolerance must be positive");

    settings::ConstraintSettings::setRattleTolerance(tolerance);
}

/**
 * @brief parsing rattle iteration
 *
 * @details default value is 20
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if iteration is negative
 */
void InputFileParserConstraints::parseRattleIteration(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw customException::InputFileException("Maximum rattle iterations must be positive");

    settings::ConstraintSettings::setRattleMaxIter(size_t(iteration));
}