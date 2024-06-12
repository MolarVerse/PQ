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

#include "inputFileParserIntegrator.hpp"

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException, customException
#include "integrator.hpp"        // for VelocityVerlet, integrator
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;

/**
 * @brief Construct a new Input File Parser Integrator:: Input File Parser
 * Integrator object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) integrator <string>
 *
 * @param engine
 */
InputFileParserIntegrator::InputFileParserIntegrator(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("integrator"),
        bind_front(&InputFileParserIntegrator::parseIntegrator, this),
        false
    );
}

/**
 * @brief Parse the integrator used in the simulation
 *
 * @details Possible options are:
 * 1) "v-verlet"  - velocity verlet integrator is used (default)
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if integrator is not valid -
 * currently only velocity verlet is supported
 */
void InputFileParserIntegrator::parseIntegrator(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto integrator = utilities::toLowerCopy(lineElements[2]);

    if (integrator == "v-verlet")
        _engine.makeIntegrator(integrator::VelocityVerlet());
    else
        throw customException::InputFileException(format(
            "Invalid integrator \"{}\" at line {} in input file",
            lineElements[2],
            lineNumber
        ));
}