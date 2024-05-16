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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"                  // for InputFileException
#include "gtest/gtest.h"                   // for Message, TestPartResult
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserThermostat.hpp"   // for InputFileParserThermostat
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "thermostatSettings.hpp"          // for ThermostatSettings
#include "throwWithMessage.hpp"            // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "temp" command
 *
 * @details if the temperature is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTemperature)
{
    EXPECT_EQ(settings::ThermostatSettings::isTemperatureSet(), false);

    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"temp", "=", "300.0"};
    parser.parseTemperature(lineElements, 0);

    EXPECT_EQ(settings::ThermostatSettings::isTemperatureSet(), true);
    EXPECT_EQ(settings::ThermostatSettings::getTargetTemperature(), 300.0);

    lineElements = {"temp", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parser.parseTemperature(lineElements, 0),
        customException::InputFileException,
        "Temperature cannot be negative"
    );
}

/**
 * @brief tests parsing the "t_relaxation" command
 *
 * @details if the relaxation time of the thermostat is negative it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRelaxationTime)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"t_relaxation", "=", "10.0"};
    parser.parseThermostatRelaxationTime(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getRelaxationTime(), 10.0);

    lineElements = {"t_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parser.parseThermostatRelaxationTime(lineElements, 0),
        customException::InputFileException,
        "Relaxation time of thermostat cannot be negative"
    );
}

/**
 * @brief tests parsing the "thermostat" command
 *
 * @details if the thermostat is not valid it throws inputFileException - valid
 * options are "none" and "berendsen"
 *
 */
TEST_F(TestInputFileReader, testParseThermostat)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"thermostat", "=", "none"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        settings::ThermostatType::NONE
    );

    lineElements = {"thermostat", "=", "berendsen"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        settings::ThermostatType::BERENDSEN
    );

    lineElements = {"thermostat", "=", "langevin"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        settings::ThermostatType::LANGEVIN
    );

    lineElements = {"thermostat", "=", "velocity_rescaling"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        settings::ThermostatType::VELOCITY_RESCALING
    );

    lineElements = {"thermostat", "=", "nh-chain"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        settings::ThermostatType::NOSE_HOOVER
    );

    lineElements = {"thermostat", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseThermostat(lineElements, 0),
        customException::InputFileException,
        "Invalid thermostat \"notValid\" at line 0 in input file. Possible "
        "options are: none, berendsen, "
        "velocity_rescaling, langevin, nh-chain"
    );
}

/**
 * @brief tests parsing the "friction" command
 *
 */
TEST_F(TestInputFileReader, testParseFriction)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"friction", "=", "0.1"};
    parser.parseThermostatFriction(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getFriction(), 0.1 * 1.0e12);

    lineElements = {"friction", "=", "-0.1"};
    EXPECT_THROW_MSG(
        parser.parseThermostatFriction(lineElements, 0),
        customException::InputFileException,
        "Friction of thermostat cannot be negative"
    );
}

/**
 * @brief tests parsing the "nh-chain-length" command
 *
 * @details if the chain length is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseChainLength)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"nh-chain-length", "=", "10"};
    parser.parseThermostatChainLength(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getNoseHooverChainLength(), 10);

    lineElements = {"nh-chain-length", "=", "-10"};
    EXPECT_THROW_MSG(
        parser.parseThermostatChainLength(lineElements, 0),
        customException::InputFileException,
        "Chain length of thermostat cannot be negative"
    );
}

/**
 * @brief tests parsing the "coupling_frequency" command
 *
 */
TEST_F(TestInputFileReader, testParseCouplingFrequency)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"coupling_frequency", "=", "10"};
    parser.parseThermostatCouplingFrequency(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getNoseHooverCouplingFrequency(),
        10
    );

    lineElements = {"coupling_frequency", "=", "-10"};
    EXPECT_THROW_MSG(
        parser.parseThermostatCouplingFrequency(lineElements, 0),
        customException::InputFileException,
        "Coupling frequency of thermostat cannot be negative"
    );
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}