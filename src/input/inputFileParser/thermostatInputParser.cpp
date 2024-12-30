/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "thermostatInputParser.hpp"

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "exceptions.hpp"           // for InputFileException, customException
#include "references.hpp"           // for References
#include "referencesOutput.hpp"     // for ReferencesOutput
#include "stringUtilities.hpp"      // for toLowerCopy
#include "thermostatSettings.hpp"   // for ThermostatSettings

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;
using namespace references;

/**
 * @brief Construct a new Input File Parser Thermostat:: Input File Parser
 * Thermostat object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) thermostat <string> 2) temp
 * <double> 3) t_relaxation <double> 4) friction <double> 5) nh-chain_length
 * <size_t> 6) coupling_frequency <double>
 *
 * @param engine
 */
ThermostatInputParser::ThermostatInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("thermostat"),
        bind_front(&ThermostatInputParser::parseThermostat, this),
        false
    );
    addKeyword(
        std::string("temp"),
        bind_front(&ThermostatInputParser::parseTemperature, this),
        false
    );
    addKeyword(
        std::string("start_temp"),
        bind_front(&ThermostatInputParser::parseStartTemperature, this),
        false
    );
    addKeyword(
        std::string("end_temp"),
        bind_front(&ThermostatInputParser::parseEndTemperature, this),
        false
    );
    addKeyword(
        std::string("temp_ramp_steps"),
        bind_front(&ThermostatInputParser::parseTemperatureRampSteps, this),
        false
    );
    addKeyword(
        std::string("temp_ramp_frequency"),
        bind_front(&ThermostatInputParser::parseTemperatureRampFrequency, this),
        false
    );
    addKeyword(
        std::string("t_relaxation"),
        bind_front(&ThermostatInputParser::parseThermostatRelaxationTime, this),
        false
    );
    addKeyword(
        std::string("friction"),
        bind_front(&ThermostatInputParser::parseThermostatFriction, this),
        false
    );
    addKeyword(
        std::string("nh-chain_length"),
        bind_front(&ThermostatInputParser::parseThermostatChainLength, this),
        false
    );
    addKeyword(
        std::string("coupling_frequency"),
        bind_front(
            &ThermostatInputParser::parseThermostatCouplingFrequency,
            this
        ),
        false
    );
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
 * @throws InputFileException if thermostat is not "none" or
 * "berendsen"
 */
void ThermostatInputParser::parseThermostat(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto thermostat = toLowerAndReplaceDashesCopy(lineElements[2]);

    using enum ThermostatType;

    if (thermostat == "none")
        ThermostatSettings::setThermostatType(NONE);

    else if (thermostat == "berendsen")
    {
        ThermostatSettings::setThermostatType(BERENDSEN);
        ReferencesOutput::addReferenceFile(_BERENDSEN_FILE_);
    }

    else if (thermostat == "velocity_rescaling" || thermostat == "rescale")
    {
        ThermostatSettings::setThermostatType(VELOCITY_RESCALING);
        ReferencesOutput::addReferenceFile(_VELOCITY_RESCALING_FILE_);
    }

    else if (thermostat == "langevin")
    {
        ThermostatSettings::setThermostatType(LANGEVIN);
        ReferencesOutput::addReferenceFile(_LANGEVIN_FILE_);
    }

    else if (thermostat == "nh_chain")
    {
        ThermostatSettings::setThermostatType(NOSE_HOOVER);
        ReferencesOutput::addReferenceFile(_NOSE_HOOVER_CHAIN_FILE_);
    }

    else
        throw InputFileException(format(
            "Invalid thermostat \"{}\" at line {} in input file.\n"
            "Possible options are: none, berendsen, velocity_rescaling, "
            "langevin, nh-chain",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parse the temperature used in the simulation
 *
 * @details Temperature is needs to be set if thermostat is not "none"
 *
 * @param lineElements
 *
 * @throws InputFileException if temperature is negative
 */
void ThermostatInputParser::parseTemperature(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto temperature = stod(lineElements[2]);

    if (temperature < 0)
        throw InputFileException("Temperature cannot be negative");

    ThermostatSettings::setTargetTemperature(temperature);
}

/**
 * @brief Parse the start temperature used in the simulation
 *
 * @details Start temperature is needs to be set if thermostat is not "none"
 *
 * @param lineElements
 *
 * @throws InputFileException if start temperature is negative
 */
void ThermostatInputParser::parseStartTemperature(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto startTemperature = stod(lineElements[2]);

    if (startTemperature < 0)
        throw InputFileException("Start temperature cannot be negative");

    ThermostatSettings::setStartTemperature(startTemperature);
}

/**
 * @brief Parse the end temperature used in the simulation
 *
 * @details End temperature is needs to be set if thermostat is not "none"
 *
 * @param lineElements
 *
 * @throws InputFileException if end temperature is negative
 */
void ThermostatInputParser::parseEndTemperature(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto endTemperature = stod(lineElements[2]);

    if (endTemperature < 0)
        throw InputFileException("End temperature cannot be negative");

    ThermostatSettings::setEndTemperature(endTemperature);
}

/**
 * @brief Parse the temperature ramp steps used in the simulation
 *
 * @details if start_temp and end_temp are set, then if temperature_ramp_steps
 * is not set, the temperature will be ramped linearly from start_temp to
 * end_temp over the full simulation time.
 *
 * @param lineElements
 *
 * @throws InputFileException if temperature ramp steps is
 * negative
 */
void ThermostatInputParser::parseTemperatureRampSteps(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto temperatureRampSteps = stoi(lineElements[2]);

    if (temperatureRampSteps < 0)
        throw InputFileException("Temperature ramp steps cannot be negative");

    ThermostatSettings::setTemperatureRampSteps(size_t(temperatureRampSteps));
}

/**
 * @brief Parse the temperature ramp frequency used in the simulation
 *
 * @details default value is 1
 *
 * @param lineElements
 *
 * @throws InputFileException if temperature ramp frequency is
 * negative
 */
void ThermostatInputParser::parseTemperatureRampFrequency(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto tempRampFreq = stoi(lineElements[2]);

    if (tempRampFreq < 0)
        throw InputFileException("Temperature ramp frequency cannot be negative"
        );

    ThermostatSettings::setTemperatureRampFrequency(size_t(tempRampFreq));
}

/**
 * @brief parses the relaxation time of the thermostat
 *
 * @details default value is 0.1
 *
 * @param lineElements
 *
 * @throws InputFileException if relaxation time is negative
 */
void ThermostatInputParser::parseThermostatRelaxationTime(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto relaxationTime = stod(lineElements[2]);

    if (relaxationTime < 0)
        throw InputFileException(
            "Relaxation time of thermostat cannot be negative"
        );

    ThermostatSettings::setRelaxationTime(relaxationTime);
}

/**
 * @brief parses the friction of the langevin thermostat
 *
 * @details default value is 1,0e11
 *
 * @param lineElements
 *
 * @throws InputFileException if friction is negative
 */
void ThermostatInputParser::parseThermostatFriction(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto friction = stod(lineElements[2]);

    if (friction < 0)
        throw InputFileException("Friction of thermostat cannot be negative");

    ThermostatSettings::setFriction(friction * 1.0e12);
}

/**
 * @brief parses the chain length of the nh-chain thermostat
 *
 * @details default value is 3
 *
 * @param lineElements
 *
 * @throws InputFileException if chain length is negative
 */
void ThermostatInputParser::parseThermostatChainLength(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto chainLength = stoi(lineElements[2]);

    if (chainLength < 0)
        throw InputFileException("Chain length of thermostat cannot be negative"
        );

    ThermostatSettings::setNoseHooverChainLength(size_t(chainLength));
}

/**
 * @brief parses the coupling frequency of the nh-chain thermostat
 *
 * @details default value is 1.0e3 cm⁻¹
 *
 * @param lineElements
 *
 * @throws InputFileException if coupling frequency is negative
 */
void ThermostatInputParser::parseThermostatCouplingFrequency(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto couplingFrequency = stod(lineElements[2]);

    if (couplingFrequency < 0)
        throw InputFileException(
            "Coupling frequency of thermostat cannot be negative"
        );

    ThermostatSettings::setNoseHooverCouplingFrequency(couplingFrequency);
}