#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace thermostat;
using namespace customException;

/**
 * @brief Parse the nonCoulombic type of the guff.dat file
 *
 * @details possible options are "none", "lj" and "buck"
 *
 * @param lineElements
 */
void InputFileReader::parseNonCoulombType(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "none")
        _engine.getSettings().setNonCoulombType("none");
    else if (lineElements[2] == "lj")
        _engine.getSettings().setNonCoulombType("lj");
    else if (lineElements[2] == "buck")
        _engine.getSettings().setNonCoulombType("buck");
    else
        throw InputFileException("Invalid nonCoulomb type \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
}