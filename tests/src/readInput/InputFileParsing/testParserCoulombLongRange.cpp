#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseCoulombLongRange)
{
    vector<string> lineElements = {"long-range", "=", "none"};
    _inputFileReader->parseCoulombLongRange(lineElements);
    EXPECT_EQ(_engine.getSettings().getCoulombLongRangeType(), "none");
    lineElements = {"long-range", "=", "wolf"};
    _inputFileReader->parseCoulombLongRange(lineElements);
    EXPECT_EQ(_engine.getSettings().getCoulombLongRangeType(), "wolf");
    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW(_inputFileReader->parseCoulombLongRange(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseWolfParameter)
{
    EXPECT_EQ(_engine.getSettings().getWolfParameter(), 0.25);
    vector<string> lineElements = {"wolf_param", "=", "1.0"};
    _inputFileReader->parseWolfParameter(lineElements);
    EXPECT_EQ(_engine.getSettings().getWolfParameter(), 1.0);
    lineElements = {"wolf_param", "=", "-1.0"};
    EXPECT_THROW(_inputFileReader->parseWolfParameter(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}