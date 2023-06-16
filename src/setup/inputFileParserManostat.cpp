#include "constants.hpp"
#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace setup;

/**
 * @brief Parse the thermostat used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseManostat(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "none")
    {
        _engine._manostat = make_unique<Manostat>();
        _engine.getSettings().setManostat("none");
    }
    else if (lineElements[2] == "berendsen")
    {
        _engine._manostat = make_unique<BerendsenManostat>();
        _engine.getSettings().setManostat("berendsen");
    }
    else
        throw InputFileException("Invalid manostat \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
}

void InputFileReader::parsePressure(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setPressure(stod(lineElements[2]));
}

void InputFileReader::parseManostatRelaxationTime(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setTauManostat(stod(lineElements[2]));
}