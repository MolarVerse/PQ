#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace thermostat;
using namespace customException;

/**
 * @brief Parse the coulombic long-range correction used in the simulation
 *
 * @details possible options are "none" and "wolf"
 *
 * @param lineElements
 */
void InputFileReader::parseCoulombLongRange(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "none")
    {
        _engine.getSettings().setCoulombLongRangeType("none");
    }
    else if (lineElements[2] == "wolf")
    {
        _engine.getSettings().setCoulombLongRangeType("wolf");
    }
    else
        throw InputFileException("Invalid long_range type for coulomb correction \"" + lineElements[2] + "\" at line " +
                                 to_string(_lineNumber) + "in input file");
}

/**
 * @brief parse the wolf parameter used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseWolfParameter(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    auto wolfParameter = stod(lineElements[2]);
    if (wolfParameter < 0.0)
        throw InputFileException("Invalid wolf parameter \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file - it has to be positive");

    _engine.getSettings().setWolfParameter(wolfParameter);
}
