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

#include "inputFileParserOptimizer.hpp"

#include <format>   // for std::format
#include <string>   // for std::string

#include "exceptions.hpp"          // for customException::InputFileException
#include "optimizerSettings.hpp"   // for OptimizerSettings
#include "stringUtilities.hpp"     // for utilities::toLowerCopy

using namespace input;
using namespace settings;

/**
 * @brief Constructor
 *
 * @details following keywords are added:
 * - optimizer <string>
 *
 * @param engine The engine
 */
InputFileParserOptimizer::InputFileParserOptimizer(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        "optimizer",
        bind_front(&InputFileParserOptimizer::parseOptimizer, this),
        false
    );
}

/**
 * @brief Parses the optimizer
 *
 * @param lineElements The elements of the line
 * @param lineNumber The line number
 */
void InputFileParserOptimizer::parseOptimizer(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto method = utilities::toLowerCopy(lineElements[2]);

    if ("gradient-descent" == method)
        OptimizerSettings::setOptimizer(Optimizer::GRADIENT_DESCENT);
    else
        throw customException::InputFileException(std::format(
            "Unknown optimizer method \"{}\" in input file "
            "at line {}.\n"
            "Possible options are: gradient-descent",
            lineElements[2],
            lineNumber
        ));
}