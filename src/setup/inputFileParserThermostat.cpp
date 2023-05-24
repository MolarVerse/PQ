#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace Setup::InputFileReader;

/**
 * @brief Parse the thermostat used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseThermostat(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "none")
    {
        _engine._thermostat = make_unique<Thermostat>();
        _engine._settings.setThermostat("none");
    }
    else if (lineElements[2] == "berendsen")
    {
        _engine._thermostat = make_unique<BerendsenThermostat>();
        _engine._settings.setThermostat("berendsen");
    }
    else
        throw InputFileException("Invalid thermostat \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file");
}

void InputFileReader::parseTemperature(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._settings.setTemperature(stod(lineElements[2]));
}

void InputFileReader::parseRelaxationTime(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine._settings.setRelaxationTime(stod(lineElements[2]));
}