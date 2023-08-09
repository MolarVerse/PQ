#include "exceptions.hpp"
#include "inputFileParser.hpp"

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
 */
void InputFileParserCellList::parseCellListActivated(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
        _engine.getCellList().activate();
    else if (lineElements[2] == "off")
    {
        // default
    }
    else
    {
        auto message =
            "Invalid cell-list keyword \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file\n";
        message += R"(Possible keywords are "on" and "off")";
        throw InputFileException(message);
    }
}

/**
 * @brief Parses the number of cells used for each dimension
 *
 * @param lineElements
 */
void InputFileParserCellList::parseNumberOfCells(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (stoi(lineElements[2]) <= 0)
        throw InputFileException("Number of cells must be positive - number of cells = " + lineElements[2]);
    _engine.getCellList().setNumberOfCells(stoul(lineElements[2]));
}