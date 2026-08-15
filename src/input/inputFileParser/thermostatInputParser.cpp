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

#include <cmath>         // for sqrt
#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <limits>        // for numeric_limits
#include <string_view>   // for string_view

#include "constants.hpp"
#include "exceptions.hpp"   // for InputFileException, customException
#include "parserUtils.hpp"
#include "references.hpp"           // for References
#include "referencesOutput.hpp"     // for ReferencesOutput
#include "stringUtilities.hpp"      // for toLowerCopy
#include "thermostatSettings.hpp"   // for ThermostatSettings

using namespace input;
using namespace customException;
using namespace settings;
using namespace utilities;
using namespace references;
using namespace constants;

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
ThermostatInputParser::ThermostatInputParser()
{
    addKeyword(
        std::string("thermostat"),
        bindMember(&ThermostatInputParser::parseThermostat, this),
        false
    );
    addKeyword(
        std::string("temp"),
        bindMember(&ThermostatInputParser::parseTemperature, this),
        false
    );
    addKeyword(
        std::string("start_temp"),
        bindMember(&ThermostatInputParser::parseStartTemperature, this),
        false
    );
    addKeyword(
        std::string("end_temp"),
        bindMember(&ThermostatInputParser::parseEndTemperature, this),
        false
    );
    addKeyword(
        std::string("temp_ramp_steps"),
        bindMember(&ThermostatInputParser::parseTemperatureRampSteps, this),
        false
    );
    addKeyword(
        std::string("temp_ramp_frequency"),
        bindMember(&ThermostatInputParser::parseTemperatureRampFrequency, this),
        false
    );
    addKeyword(
        std::string("t_relaxation"),
        bindMember(&ThermostatInputParser::parseThermostatRelaxationTime, this),
        false
    );
    addKeyword(
        std::string("friction"),
        bindMember(&ThermostatInputParser::parseThermostatFriction, this),
        false
    );
    addKeyword(
        std::string("nh-chain_length"),
        bindMember(&ThermostatInputParser::parseThermostatChainLength, this),
        false
    );
    addKeyword(
        std::string("coupling_frequency"),
        bindMember(
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
        ReferencesOutput::addReferenceFile(BERENDSEN_FILE);
    }

    else if (thermostat == "velocity_rescaling" || thermostat == "rescale")
    {
        ThermostatSettings::setThermostatType(VELOCITY_RESCALING);
        ReferencesOutput::addReferenceFile(VELOCITY_RESCALING_FILE);
    }

    else if (thermostat == "langevin")
    {
        ThermostatSettings::setThermostatType(LANGEVIN);
        ReferencesOutput::addReferenceFile(LANGEVIN_FILE);
    }

    else if (thermostat == "nh_chain")
    {
        ThermostatSettings::setThermostatType(NOSE_HOOVER);
        ReferencesOutput::addReferenceFile(NOSE_HOOVER_CHAIN_FILE);
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

    const auto temperature = stringToFiniteDouble(lineElements[2]);

    if (temperature < 0.0)
        throw InputFileException("Temperature must be finite and non-negative");

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

    const auto startTemperature = stringToFiniteDouble(lineElements[2]);

    if (startTemperature < 0.0)
        throw InputFileException(
            "Start temperature must be finite and non-negative"
        );

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

    const auto endTemperature = stringToFiniteDouble(lineElements[2]);

    if (endTemperature < 0.0)
        throw InputFileException(
            "End temperature must be finite and non-negative"
        );

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

    const auto temperatureRampSteps = stringToInt(lineElements[2]);

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

    const auto tempRampFreq = stringToInt(lineElements[2]);

    if (tempRampFreq < 1)
        throw InputFileException(
            "Temperature ramp frequency must be greater than zero"
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

    const auto relaxationTime = stringToFiniteDouble(lineElements[2]);

    if (relaxationTime <= 0.0)
        throw InputFileException(
            "Relaxation time of thermostat must be finite and greater than zero"
        );

    if (relaxationTime > std::numeric_limits<double>::max() / PS_TO_FS)
        throw InputFileException(
            "Relaxation time of thermostat is too large to represent in "
            "femtoseconds"
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

    const auto friction = stringToFiniteDouble(lineElements[2]);

    if (friction < 0.0)
        throw InputFileException(
            "Friction of thermostat must be finite and non-negative"
        );

    if (friction >
        std::numeric_limits<double>::max() / defaults::MAX_FRICTION_CONVERSION)
        throw InputFileException(
            "Friction of thermostat is too large to represent in inverse "
            "seconds"
        );

    ThermostatSettings::setFriction(
        friction * constants::NOSE_HOVER_FRICTION_INPUT_TO_INTERNAL
    );
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

    const auto chainLength = stringToInt(lineElements[2]);

    if (chainLength < 1)
        throw InputFileException(
            "Chain length of thermostat must be greater than zero"
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

    const auto couplingFrequency = stringToFiniteDouble(lineElements[2]);

    if (couplingFrequency < 0.0)
        throw InputFileException(
            "Coupling frequency of thermostat must be finite and non-negative"
        );

    if (couplingFrequency >
        std::sqrt(std::numeric_limits<double>::max()) / PER_CM_TO_HZ)
        throw InputFileException(
            "Coupling frequency of thermostat is too large to represent in "
            "hertz"
        );

    ThermostatSettings::setNoseHooverCouplingFrequency(couplingFrequency);
}
