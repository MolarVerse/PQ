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

#include "inputFileParserThermostat.hpp"

#include "exceptions.hpp"           // for InputFileException, customException
#include "references.hpp"           // for References
#include "referencesOutput.hpp"     // for ReferencesOutput
#include "stringUtilities.hpp"      // for toLowerCopy
#include "thermostatSettings.hpp"   // for ThermostatSettings

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace input;

/**
 * @brief Construct a new Input File Parser Thermostat:: Input File Parser Thermostat object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) thermostat <string>
 * 2) temp <double>
 * 3) t_relaxation <double>
 * 4) friction <double>
 * 5) nh-chain_length <size_t>
 * 6) coupling_frequency <double>
 *
 * @param engine
 */
InputFileParserThermostat::InputFileParserThermostat(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("thermostat"), bind_front(&InputFileParserThermostat::parseThermostat, this), false);
    addKeyword(std::string("temp"), bind_front(&InputFileParserThermostat::parseTemperature, this), false);
    addKeyword(std::string("t_relaxation"), bind_front(&InputFileParserThermostat::parseThermostatRelaxationTime, this), false);
    addKeyword(std::string("friction"), bind_front(&InputFileParserThermostat::parseThermostatFriction, this), false);
    addKeyword(std::string("nh-chain_length"), bind_front(&InputFileParserThermostat::parseThermostatChainLength, this), false);
    addKeyword(
        std::string("coupling_frequency"), bind_front(&InputFileParserThermostat::parseThermostatCouplingFrequency, this), false);
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

    const auto thermostat = utilities::toLowerCopy(lineElements[2]);

    if (thermostat == "none")
        settings::ThermostatSettings::setThermostatType("none");

    else if (thermostat == "berendsen")
    {
        settings::ThermostatSettings::setThermostatType("berendsen");
        references::ReferencesOutput::addReferenceFile(references::_BERENDSEN_FILE_);
    }

    else if (thermostat == "velocity_rescaling" || thermostat == "rescale")
    {
        settings::ThermostatSettings::setThermostatType("velocity_rescaling");
        references::ReferencesOutput::addReferenceFile(references::_VELOCITY_RESCALING_FILE_);
    }

    else if (thermostat == "langevin")
    {
        settings::ThermostatSettings::setThermostatType("langevin");
        references::ReferencesOutput::addReferenceFile(references::_LANGEVIN_FILE_);
    }

    else if (thermostat == "nh-chain")
    {
        settings::ThermostatSettings::setThermostatType("nh-chain");
        references::ReferencesOutput::addReferenceFile(references::_NOSE_HOOVER_CHAIN_FILE_);
    }

    else
        throw customException::InputFileException(format("Invalid thermostat \"{}\" at line {} in input file. Possible options "
                                                         "are: none, berendsen, velocity_rescaling, langevin, nh-chain",
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

/**
 * @brief parses the chain length of the nh-chain thermostat
 *
 * @details default value is 3
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if chain length is negative
 */
void InputFileParserThermostat::parseThermostatChainLength(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto chainLength = stoi(lineElements[2]);

    if (chainLength < 0)
        throw customException::InputFileException("Chain length of thermostat cannot be negative");

    settings::ThermostatSettings::setNoseHooverChainLength(size_t(chainLength));
}

/**
 * @brief parses the coupling frequency of the nh-chain thermostat
 *
 * @details default value is 1.0e3 cm⁻¹
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if coupling frequency is negative
 */
void InputFileParserThermostat::parseThermostatCouplingFrequency(const std::vector<std::string> &lineElements,
                                                                 const size_t                    lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto couplingFrequency = stod(lineElements[2]);

    if (couplingFrequency < 0)
        throw customException::InputFileException("Coupling frequency of thermostat cannot be negative");

    settings::ThermostatSettings::setNoseHooverCouplingFrequency(couplingFrequency);
}