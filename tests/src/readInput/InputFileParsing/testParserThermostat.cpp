#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseTemperature)
{
    vector<string> lineElements = {"temperature", "=", "300.0"};
    _inputFileReader->parseTemperature(lineElements);
    EXPECT_EQ(_engine.getSettings().getTemperature(), 300.0);
    lineElements = {"temperature", "=", "-100.0"};
    EXPECT_THROW(_inputFileReader->parseTemperature(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseRelaxationTime)
{
    vector<string> lineElements = {"temperature", "=", "0.1"};
    _inputFileReader->parseThermostatRelaxationTime(lineElements);
    EXPECT_EQ(_engine.getSettings().getRelaxationTime(), 0.1);
    lineElements = {"temperature", "=", "-100.0"};
    EXPECT_THROW(_inputFileReader->parseTemperature(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseThermostat)
{
    vector<string> lineElements = {"thermostat", "=", "none"};
    _inputFileReader->parseThermostat(lineElements);
    EXPECT_EQ(_engine.getSettings().getThermostat(), "none");
    lineElements = {"thermostat", "=", "berendsen"};
    _inputFileReader->parseThermostat(lineElements);
    EXPECT_EQ(_engine.getSettings().getThermostat(), "berendsen");
    lineElements = {"thermostat", "=", "notvalid"};
    EXPECT_THROW(_inputFileReader->parseThermostat(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}