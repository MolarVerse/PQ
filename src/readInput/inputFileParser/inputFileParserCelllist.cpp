#include "inputFileParserCellList.hpp"   // for InputFileParserCellList

#include "celllist.hpp"          // for CellList
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException
#include "inputFileParser.hpp"   // for checkCommand, InputFileParser
#include "stringUtilities.hpp"   // for toLowerCopy

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front
#include <string>       // for allocator, operator==, string
#include <vector>       // for vector

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Cell List:: Input File Parser Cell List object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) cell-list <on/off>
 * 2) cell-number <size_t>
 *
 * @param engine
 */
InputFileParserCellList::InputFileParserCellList(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("cell-list"), bind_front(&InputFileParserCellList::parseCellListActivated, this), false);
    addKeyword(std::string("cell-number"), bind_front(&InputFileParserCellList::parseNumberOfCells, this), false);
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
 * @throws customException::InputFileException if cell-list keyword is not "on" or "off"
 */
void InputFileParserCellList::parseCellListActivated(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto cellListActivated = utilities::toLowerCopy(lineElements[2]);

    if (cellListActivated == "on")
        _engine.getCellList().activate();

    else if (cellListActivated == "off")
        _engine.getCellList().deactivate();

    else
        throw customException::InputFileException(
            format(R"(Invalid cell-list keyword "{}" at line {} in input file\n Possible keywords are "on" and "off")",
                   lineElements[2],
                   lineNumber));
}

/**
 * @brief Parses the number of cells used for each dimension
 *
 * @details default value is 7
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if number of cells is not positive
 */
void InputFileParserCellList::parseNumberOfCells(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto cellNumber = stoi(lineElements[2]);

    if (cellNumber <= 0)
        throw customException::InputFileException("Number of cells must be positive - number of cells = " + lineElements[2]);

    _engine.getCellList().setNumberOfCells(size_t(cellNumber));
}