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

#include "constraintsInputParser.hpp"

#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "constraints.hpp"          // for Constraints
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for InputFileException

using namespace input;
using namespace engine;
using namespace settings;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Constraints:: Input File Parser
 * Constraints object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) shake <on/off> 2)
 * shake-tolerance <double> 3) shake-iter <size_t> 4) rattle-iter <size_t> 5)
 * rattle-tolerance <double>
 *
 * @param engine
 */
ConstraintsInputParser::ConstraintsInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("shake"),
        bind_front(&ConstraintsInputParser::parseShakeActivated, this),
        false
    );
    addKeyword(
        std::string("shake-tolerance"),
        bind_front(&ConstraintsInputParser::parseShakeTolerance, this),
        false
    );
    addKeyword(
        std::string("shake-iter"),
        bind_front(&ConstraintsInputParser::parseShakeIteration, this),
        false
    );
    addKeyword(
        std::string("rattle-iter"),
        bind_front(&ConstraintsInputParser::parseRattleIteration, this),
        false
    );
    addKeyword(
        std::string("rattle-tolerance"),
        bind_front(&ConstraintsInputParser::parseRattleTolerance, this),
        false
    );

    addKeyword(
        std::string("distance-constraints"),
        bind_front(
            &ConstraintsInputParser::parseDistanceConstraintActivated,
            this
        ),
        false
    );
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
 * @throws InputFileException if keyword is not valid -
 * currently only on and off are supported
 */
void ConstraintsInputParser::parseShakeActivated(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    auto &constraints = _engine.getConstraints();

    if (lineElements[2] == "on" || lineElements[2] == "shake")
    {
        constraints.activateShake();
        ConstraintSettings::activateShake();
    }
    else if (lineElements[2] == "off")
    {
        constraints.deactivateShake();
        ConstraintSettings::deactivateShake();
    }
    else if (lineElements[2] == "mshake")
    {
        constraints.activateMShake();
        constraints.activateShake();
        ConstraintSettings::activateMShake();
        ConstraintSettings::activateShake();
    }
    else
    {
        auto message = format(
            "Invalid shake keyword \"{}\" at line {} in input file\n"
            "Possible keywords are: \"on\", \"off\", \"shake\", \"mshake\"",
            lineElements[2],
            lineNumber
        );

        throw InputFileException(message);
    }
}

/**
 * @brief parsing shake tolerance
 *
 * @details default value is 1e-8
 *
 * @param lineElements
 *
 * @throw InputFileException if tolerance is negative
 */
void ConstraintsInputParser::parseShakeTolerance(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw InputFileException("Shake tolerance must be positive");

    ConstraintSettings::setShakeTolerance(tolerance);
}

/**
 * @brief parsing shake iteration
 *
 * @details default value is 20
 *
 * @param lineElements
 *
 * @throw InputFileException if iteration is negative
 */
void ConstraintsInputParser::parseShakeIteration(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw InputFileException("Maximum shake iterations must be positive");

    ConstraintSettings::setShakeMaxIter(size_t(iteration));
}

/**
 * @brief parsing rattle tolerance
 *
 * @details default value is 1e-8
 *
 * @param lineElements
 *
 * @throw InputFileException if tolerance is negative
 */
void ConstraintsInputParser::parseRattleTolerance(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw InputFileException("Rattle tolerance must be positive");

    ConstraintSettings::setRattleTolerance(tolerance);
}

/**
 * @brief parsing rattle iteration
 *
 * @details default value is 20
 *
 * @param lineElements
 *
 * @throw InputFileException if iteration is negative
 */
void ConstraintsInputParser::parseRattleIteration(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw InputFileException("Maximum rattle iterations must be positive");

    ConstraintSettings::setRattleMaxIter(size_t(iteration));
}

/**
 * @brief parsing if distance constraint is activated
 *
 * @details Possible options are:
 * 1) "on"  - distance constraint is activated
 * 2) "off" - distance constraint is deactivated (default)
 *
 * @param lineElements
 *
 * @throws InputFileException if keyword is not valid -
 * currently only on and off are supported
 */
void ConstraintsInputParser::parseDistanceConstraintActivated(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    if (lineElements[2] == "on")
    {
        _engine.getConstraints().activateDistanceConstraints();
        ConstraintSettings::activateDistanceConstraints();
    }
    else if (lineElements[2] == "off")
    {
        _engine.getConstraints().deactivateDistanceConstraints();
        ConstraintSettings::deactivateDistanceConstraints();
    }
    else
    {
        auto message = format(
            "Invalid {} keyword \"{}\" at line {} in input file\n"
            "Possible keywords are \"on\" and \"off\"",
            lineElements[0],
            lineElements[2],
            lineNumber
        );
        throw InputFileException(message);
    }
}