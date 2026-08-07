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

#include "cellListInputParser.hpp"

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front
#include <string>       // for allocator, operator==, string
#include <vector>       // for vector

#include "celllist.hpp"          // for CellList
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException
#include "inputFileParser.hpp"   // for checkCommand, InputFileParser
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace input;
using namespace engine;
using namespace utilities;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Cell List:: Input File Parser Cell
 * List object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) cell-list <on/off> 2)
 * cell-number <size_t>
 *
 * @param engine
 */
CellListInputParser::CellListInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("cell-list"),
        bind_front(&CellListInputParser::parseCellListActivated, this),
        false
    );
    addKeyword(
        std::string("cell-number"),
        bind_front(&CellListInputParser::parseNumberOfCells, this),
        false
    );
}

/**
 * @brief Parses if cell-list should be used in simulation
 *
 * @details Possible options are:
 * 1) "on"  - cell-list is activated
 * 2) "off" - cell-list is deactivated (default)
 *
 * @param lineElements
 *
 * @throws InputFileException if cell-list keyword is not "on"
 * or "off"
 */
void CellListInputParser::parseCellListActivated(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto cellListActivated = toLowerCopy(lineElements[2]);

    if (cellListActivated == "on")
        _engine.getCellList().activate();

    else if (cellListActivated == "off")
        _engine.getCellList().deactivate();

    else
        throw InputFileException(std::format(
            "Invalid cell-list keyword \"{}\" "
            "at line {} in input file\n"
            "Possible keywords are \"on\" and \"off\"",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parses the number of cells used for each dimension
 *
 * @details default value is 7
 *
 * @param lineElements
 *
 * @throws InputFileException if number of cells is not
 * positive
 */
void CellListInputParser::parseNumberOfCells(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto cellNumber = stoi(lineElements[2]);

    if (cellNumber <= 0)
        throw InputFileException(
            "Number of cells must be positive - number of cells = " +
            lineElements[2]
        );

    _engine.getCellList().setNumberOfCells(size_t(cellNumber));
}