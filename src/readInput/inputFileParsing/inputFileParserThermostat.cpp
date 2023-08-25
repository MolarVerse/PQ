#include "inputFileParserThermostat.hpp"

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, customException
#include "settings.hpp"     // for Settings
#include "thermostat.hpp"   // for BerendsenThermostat, Thermostat, thermostat

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

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
        throw InputFileException(format("Invalid thermostat \"{}\" at line {} in input file", lineElements[2], lineNumber));
}

/**
 * @brief Parse the temperature used in the simulation
 *
 * @param lineElements
 *
 * @throws InputFileException if temperature is negative
 */
void InputFileParserThermostat::parseTemperature(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto temperature = stod(lineElements[2]);

    if (temperature < 0)
        throw InputFileException("Temperature cannot be negative");

    _engine.getSettings().setTemperature(temperature);
}

/**
 * @brief parses the relaxation time of the thermostat
 *
 * @param lineElements
 *
 * @throws InputFileException if relaxation time is negative
 */
void InputFileParserThermostat::parseThermostatRelaxationTime(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    const auto relaxationTime = stod(lineElements[2]);

    if (relaxationTime < 0)
        throw InputFileException("Relaxation time of thermostat cannot be negative");

    _engine.getSettings().setRelaxationTime(relaxationTime);
}