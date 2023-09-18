#include "inputFileParserThermostat.hpp"

#include "exceptions.hpp"           // for InputFileException, customException
#include "thermostatSettings.hpp"   // for ThermostatSettings

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Thermostat:: Input File Parser Thermostat object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) thermostat <string>
 * 2) temp <double>
 * 3) t_relaxation <double>
 * 4) friction <double>
 *
 * @param engine
 */
InputFileParserThermostat::InputFileParserThermostat(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("thermostat"), bind_front(&InputFileParserThermostat::parseThermostat, this), false);
    addKeyword(std::string("temp"), bind_front(&InputFileParserThermostat::parseTemperature, this), false);
    addKeyword(std::string("t_relaxation"), bind_front(&InputFileParserThermostat::parseThermostatRelaxationTime, this), false);
    addKeyword(std::string("friction"), bind_front(&InputFileParserThermostat::parseThermostatFriction, this), false);
}

/**
 * @brief Parse the thermostat used in the simulation
 *
 * @details Possible options are:
 * 1) none               - no thermostat (default)
 * 2) berendsen          - berendsen thermostat
 * 3) velocity_rescaling - velocity rescaling thermostat
 * 4) langevin           - langevin thermostat
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if thermostat is not "none" or "berendsen"
 */
void InputFileParserThermostat::parseThermostat(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    if (lineElements[2] == "none")
        settings::ThermostatSettings::setThermostatType("none");
    else if (lineElements[2] == "berendsen")
        settings::ThermostatSettings::setThermostatType("berendsen");
    else if (lineElements[2] == "velocity_rescaling")
        settings::ThermostatSettings::setThermostatType("velocity_rescaling");
    else if (lineElements[2] == "langevin")
        settings::ThermostatSettings::setThermostatType("langevin");
    else
        throw customException::InputFileException(format("Invalid thermostat \"{}\" at line {} in input file. Possible options "
                                                         "are: none, berendsen, velocity_rescaling, langevin",
                                                         lineElements[2],
                                                         lineNumber));
}

/**
 * @brief Parse the temperature used in the simulation
 *
 * @details Temperature is needs to be set if thermostat is not "none"
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if temperature is negative
 */
void InputFileParserThermostat::parseTemperature(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto temperature = stod(lineElements[2]);

    if (temperature < 0)
        throw customException::InputFileException("Temperature cannot be negative");

    settings::ThermostatSettings::setTemperatureSet(true);
    settings::ThermostatSettings::setTargetTemperature(temperature);
}

/**
 * @brief parses the relaxation time of the thermostat
 *
 * @details default value is 0.1
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if relaxation time is negative
 */
void InputFileParserThermostat::parseThermostatRelaxationTime(const std::vector<std::string> &lineElements,
                                                              const size_t                    lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto relaxationTime = stod(lineElements[2]);

    if (relaxationTime < 0)
        throw customException::InputFileException("Relaxation time of thermostat cannot be negative");

    settings::ThermostatSettings::setRelaxationTime(relaxationTime);
}

/**
 * @brief parses the friction of the langevin thermostat
 *
 * @details default value is 1,0e11
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if friction is negative
 */
void InputFileParserThermostat::parseThermostatFriction(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto friction = stod(lineElements[2]);

    if (friction < 0)
        throw customException::InputFileException("Friction of thermostat cannot be negative");

    settings::ThermostatSettings::setFriction(friction * 1.0e12);
}