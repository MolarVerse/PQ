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

#include "exceptions.hpp"            // for InputFileException
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "thermostatInputParser.hpp"
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "throwWithMessage.hpp"     // for EXPECT_THROW_MSG

/**
 * @brief tests parsing the "temp" command
 *
 * @details if the temperature is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTemperature)
{
    EXPECT_EQ(settings::ThermostatSettings::isTemperatureSet(), false);

    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("temp"));
    const auto& parseFunc = funcMap.at("temp");

    std::vector<std::string> lineElements = {"temp", "=", "300.0"};
    parseFunc(lineElements, 0);

    EXPECT_EQ(settings::ThermostatSettings::isTemperatureSet(), true);
    EXPECT_EQ(settings::ThermostatSettings::getTargetTemperature(), 300.0);

    _clearParser(parser);

    lineElements = {"temp", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100.0\" for key \"temp\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "0"
    );

    _clearParser(parser);

    lineElements = {"temp", "=", "0"};
    EXPECT_NO_THROW(parseFunc(lineElements, 0));
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
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("t_relaxation"));
    const auto& parseFunc = funcMap.at("t_relaxation");

    std::vector<std::string> lineElements = {"t_relaxation", "=", "10.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getRelaxationTime(), 10.0);

    _clearParser(parser);

    lineElements = {"t_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100.0\" for key \"t_relaxation\" at line 0 in input "
        "file: failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
    );

    _clearParser(parser);

    lineElements = {"t_relaxation", "=", "1e308"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1e308\" for key \"t_relaxation\" at line 0 in input "
        "file: failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
    );

    _clearParser(parser);

    lineElements = {"t_relaxation", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"t_relaxation\" at line 0 in input file: "
        "failed validation with message Value must be between 0 and "
        "1.7976931348623156e+305"
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
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("thermostat"));
    const auto& parseFunc = funcMap.at("thermostat");

    std::vector<std::string> lineElements = {"thermostat", "=", "none"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::NONE
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "berendsen"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::BERENDSEN
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "langevin"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::LANGEVIN
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "velocity_rescaling"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "rescale"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "nh-chain"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::NOSE_HOOVER
    );

    _clearParser(parser);

    lineElements = {"thermostat", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"thermostat\" at line 0 in input "
        "file. Allowed values: none, berendsen, velocity_rescaling, langevin, "
        "nose_hoover, nh_chain, rescale"
    );
}

/**
 * @brief tests parsing the "friction" command
 *
 */
TEST_F(TestInputFileReader, testParseFriction)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("friction"));
    const auto& parseFunc = funcMap.at("friction");

    std::vector<std::string> lineElements = {"friction", "=", "0.1"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getFriction(), 0.1 * 1.0e12);

    _clearParser(parser);

    lineElements = {"friction", "=", "-0.1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-0.1\" for key \"friction\" at line 0 in input file: "
        "failed validation with message Value must be between 0 and "
        "1.797693134862316e+296"
    );

    _clearParser(parser);

    lineElements = {"friction", "=", "1e308"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1e308\" for key \"friction\" at line 0 in input file: "
        "failed validation with message Value must be between 0 and "
        "1.797693134862316e+296"
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
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("nh_chain_length"));
    const auto& parseFunc = funcMap.at("nh_chain_length");

    std::vector<std::string> lineElements = {"nh-chain-length", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getNoseHooverChainLength(), 10);

    _clearParser(parser);

    lineElements = {"nh-chain-length", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"nh_chain_length\" at line 0 in input "
        "file. Value must be a positive integer"
    );

    _clearParser(parser);

    lineElements = {"nh-chain-length", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"nh_chain_length\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 1"
    );
}

/**
 * @brief tests parsing the "coupling_frequency" command
 *
 */
TEST_F(TestInputFileReader, testParseCouplingFrequency)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("coupling_frequency"));
    const auto& parseFunc = funcMap.at("coupling_frequency");

    std::vector<std::string> lineElements = {"coupling_frequency", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::ThermostatSettings::getNoseHooverCouplingFrequency(),
        10
    );

    _clearParser(parser);

    lineElements = {"coupling_frequency", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"coupling_frequency\" at line 0 in "
        "input file: failed validation with message Value must be between 0 "
        "and 4.47236332074191e+143"
    );

    _clearParser(parser);

    lineElements = {"coupling_frequency", "=", "1e308"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1e308\" for key \"coupling_frequency\" at line 0 in "
        "input file: failed validation with message Value must be between 0 "
        "and 4.47236332074191e+143"
    );
}

/**
 * @brief tests parsing the "temp_ramp_steps" command
 *
 * @details if the number of steps is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTemperatureRampSteps)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("temp_ramp_steps"));
    const auto& parseFunc = funcMap.at("temp_ramp_steps");

    std::vector<std::string> lineElements = {"temp_ramp_steps", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getTemperatureRampSteps(), 10);

    _clearParser(parser);

    lineElements = {"temp_ramp_steps", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"temp_ramp_steps\" at line 0 in input "
        "file. Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "temp_ramp_frequency" command
 *
 * @details if the frequency is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTemperatureRampFrequency)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("temp_ramp_frequency"));
    const auto& parseFunc = funcMap.at("temp_ramp_frequency");

    std::vector<std::string> lineElements = {"temp_ramp_frequency", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getTemperatureRampFrequency(), 10);

    _clearParser(parser);

    lineElements = {"temp_ramp_frequency", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"temp_ramp_frequency\" at line 0 in "
        "input file. Value must be a positive integer"
    );

    _clearParser(parser);

    lineElements = {"temp_ramp_frequency", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"temp_ramp_frequency\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    );
}

/**
 * @brief tests parsing the "start_temperature" command
 *
 * @details if the start temperature is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseStartTemperature)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("start_temp"));
    const auto& parseFunc = funcMap.at("start_temp");

    std::vector<std::string> lineElements = {"start_temp", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getStartTemperature(), 10);

    _clearParser(parser);

    lineElements = {"start_temp", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"start_temp\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "or equal to 0"
    );

    _clearParser(parser);

    lineElements = {"start_temp", "=", "0"};
    EXPECT_NO_THROW(parseFunc(lineElements, 0));
}

/**
 * @brief tests parsing the "end_temperature" command
 *
 * @details if the end temperature is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseEndTemperature)
{
    input::ThermostatInputParser parser;
    const auto                   funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("end_temperature"));
    const auto& parseFunc = funcMap.at("end_temperature");

    std::vector<std::string> lineElements = {"end_temperature", "=", "10"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getEndTemperature(), 10);

    _clearParser(parser);

    lineElements = {"end_temperature", "=", "-10"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-10\" for key \"end_temperature\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 0"
    );

    _clearParser(parser);

    lineElements = {"end_temp", "=", "0"};
    EXPECT_NO_THROW(parseFunc(lineElements, 0));
}
