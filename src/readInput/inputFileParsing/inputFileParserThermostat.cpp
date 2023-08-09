#include "inputFileParser.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace thermostat;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Thermostat:: Input File Parser Thermostat object
 *
 * @param engine
 */
InputFileParserThermostat::InputFileParserThermostat(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("thermostat"), bind_front(&InputFileParserThermostat::parseThermostat, this), false);
    addKeyword(string("temp"), bind_front(&InputFileParserThermostat::parseTemperature, this), false);
    addKeyword(string("t_relaxation"), bind_front(&InputFileParserThermostat::parseThermostatRelaxationTime, this), false);
}

/**
 * @brief Parse the thermostat used in the simulation
 *
 * @param lineElements
 */
void InputFileParserThermostat::parseThermostat(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "none")
    {
        _engine.makeThermostat(Thermostat());
        _engine.getSettings().setThermostat("none");
    }
    else if (lineElements[2] == "berendsen")
    {
        _engine.makeThermostat(BerendsenThermostat());
        _engine.getSettings().setThermostat("berendsen");
    }
    else
        throw InputFileException("Invalid thermostat \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                                 "in input file");
}

/**
 * @brief Parse the temperature used in the simulation
 *
 * @param lineElements
 */
void InputFileParserThermostat::parseTemperature(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setTemperature(stod(lineElements[2]));
}

/**
 * @brief parses the relaxation time of the thermostat
 *
 * @param lineElements
 */
void InputFileParserThermostat::parseThermostatRelaxationTime(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setRelaxationTime(stod(lineElements[2]));
}