#include "exceptions.hpp"                  // for InputFileException
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserThermostat.hpp"   // for InputFileParserThermostat
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "thermostatSettings.hpp"          // for ThermostatSettings
#include "throwWithMessage.hpp"            // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput;

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
        parser.parseTemperature(lineElements, 0), customException::InputFileException, "Temperature cannot be negative");
}

/**
 * @brief tests parsing the "t_relaxation" command
 *
 * @details if the relaxation time of the thermostat is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRelaxationTime)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"t_relaxation", "=", "10.0"};
    parser.parseThermostatRelaxationTime(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getRelaxationTime(), 10.0);

    lineElements = {"t_relaxation", "=", "-100.0"};
    EXPECT_THROW_MSG(parser.parseThermostatRelaxationTime(lineElements, 0),
                     customException::InputFileException,
                     "Relaxation time of thermostat cannot be negative");
}

/**
 * @brief tests parsing the "thermostat" command
 *
 * @details if the thermostat is not valid it throws inputFileException - valid options are "none" and "berendsen"
 *
 */
TEST_F(TestInputFileReader, testParseThermostat)
{
    InputFileParserThermostat parser(*_engine);
    std::vector<std::string>  lineElements = {"thermostat", "=", "none"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getThermostatType(), settings::ThermostatType::NONE);

    lineElements = {"thermostat", "=", "berendsen"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getThermostatType(), settings::ThermostatType::BERENDSEN);

    lineElements = {"thermostat", "=", "langevin"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getThermostatType(), settings::ThermostatType::LANGEVIN);

    lineElements = {"thermostat", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseThermostat(lineElements, 0),
                     customException::InputFileException,
                     "Invalid thermostat \"notValid\" at line 0 in input file. Possible options are: none, berendsen, langevin");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}