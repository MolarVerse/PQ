#include "exceptions.hpp"
#include "inputFileParserThermostat.hpp"
#include "testInputFileReader.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "temp" command
 *
 * @details if the temperature is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTemperature)
{
    EXPECT_EQ(settings::ThermostatSettings::isTemperatureSet(), false);

    InputFileParserThermostat parser(_engine);
    vector<string>            lineElements = {"temp", "=", "300.0"};
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
    InputFileParserThermostat parser(_engine);
    vector<string>            lineElements = {"t_relaxation", "=", "10.0"};
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
    InputFileParserThermostat parser(_engine);
    vector<string>            lineElements = {"thermostat", "=", "none"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getThermostatType(), "none");

    lineElements = {"thermostat", "=", "berendsen"};
    parser.parseThermostat(lineElements, 0);
    EXPECT_EQ(settings::ThermostatSettings::getThermostatType(), "berendsen");

    lineElements = {"thermostat", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseThermostat(lineElements, 0),
                     customException::InputFileException,
                     "Invalid thermostat \"notValid\" at line 0 in input file");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}