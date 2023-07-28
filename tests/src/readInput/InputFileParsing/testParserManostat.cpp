#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParsePressure)
{
    vector<string> lineElements = {"pressure", "=", "300.0"};
    _inputFileReader->parsePressure(lineElements);
    EXPECT_EQ(_engine.getSettings().getPressure(), 300.0);
}

TEST_F(TestInputFileReader, testParseRelaxationTimeManostat)
{
    vector<string> lineElements = {"temperature", "=", "0.1"};
    _inputFileReader->parseManostatRelaxationTime(lineElements);
    EXPECT_EQ(_engine.getSettings().getTauManostat(), 0.1);
    lineElements = {"temperature", "=", "-100.0"};
    EXPECT_THROW(_inputFileReader->parseManostatRelaxationTime(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseManostat)
{
    vector<string> lineElements = {"thermostat", "=", "none"};
    _inputFileReader->parseManostat(lineElements);
    EXPECT_EQ(_engine.getSettings().getManostat(), "none");
    lineElements = {"Manostat", "=", "berendsen"};
    _inputFileReader->parseManostat(lineElements);
    EXPECT_EQ(_engine.getSettings().getManostat(), "berendsen");
    lineElements = {"Manostat", "=", "notvalid"};
    EXPECT_THROW(_inputFileReader->parseManostat(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}