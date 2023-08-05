#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Parse the integrator used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseForceFieldType(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "on")
    {
        _engine.getForceFieldPtr()->activate();
        _engine.getForceFieldPtr()->activateNonCoulombic();
    }
    else if (lineElements[2] == "off")
    {
        _engine.getForceFieldPtr()->deactivate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else if (lineElements[2] == "bonded")
    {
        _engine.getForceFieldPtr()->activate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else
        throw InputFileException("Invalid force-field keyword \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file - possible keywords are \"on\", \"off\" or \"bonded\"");
}