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

#include "inputFileParserRingPolymer.hpp"

#include "exceptions.hpp"            // for InputFileException, customException
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace input;

/**
 * @brief Construct a new InputFileParserRingPolymer:: InputFileParserRingPolymer object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) rpmd_n_replica <size_t>
 *
 * @param engine
 */
InputFileParserRingPolymer::InputFileParserRingPolymer(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("rpmd_n_replica"), bind_front(&InputFileParserRingPolymer::parseNumberOfBeads, this), false);
}

/**
 * @brief parse number of beads for ring polymer md
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserRingPolymer::parseNumberOfBeads(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto numberOfBeads = stoi(lineElements[2]);

    if (numberOfBeads < 2)
        throw customException::InputFileException(
            std::format("Number of beads must be at least 2 - in input file in line {}", lineNumber));

    settings::RingPolymerSettings::setNumberOfBeads(size_t(numberOfBeads));
}