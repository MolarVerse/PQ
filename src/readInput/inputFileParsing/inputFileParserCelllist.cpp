#include "inputFileParserCellList.hpp"

#include "exceptions.hpp"

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Cell List:: Input File Parser Cell List object
 *
 * @param engine
 */
InputFileParserCellList::InputFileParserCellList(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("cell-list"), bind_front(&InputFileParserCellList::parseCellListActivated, this), false);
    addKeyword(string("cell-number"), bind_front(&InputFileParserCellList::parseNumberOfCells, this), false);
}

/**
 * @brief Parses if cell-list should be used in simulation
 *
 * @param lineElements
 *
 * @throws InputFileException if cell-list keyword is not "on" or "off"
 */
void InputFileParserCellList::parseCellListActivated(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
        _engine.getCellList().activate();
    else if (lineElements[2] == "off")
        _engine.getCellList().deactivate();
    else
        throw InputFileException(
            format(R"(Invalid cell-list keyword "{}" at line {} in input file\n Possible keywords are "on" and "off")",
                   lineElements[2],
                   lineNumber));
}

/**
 * @brief Parses the number of cells used for each dimension
 *
 * @param lineElements
 *
 * @throws InputFileException if number of cells is not positive
 */
void InputFileParserCellList::parseNumberOfCells(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    if (stoi(lineElements[2]) <= 0)
        throw InputFileException("Number of cells must be positive - number of cells = " + lineElements[2]);

    _engine.getCellList().setNumberOfCells(stoul(lineElements[2]));
}