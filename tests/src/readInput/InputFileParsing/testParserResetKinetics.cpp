#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseNScale)
{
    vector<string> lineElements = {"nscale", "=", "3"};
    _inputFileReader->parseNScale(lineElements);
    EXPECT_EQ(_engine.getSettings().getNScale(), 3);
    lineElements = {"temperature", "=", "-1"};
    EXPECT_THROW(_inputFileReader->parseNScale(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseFScale)
{
    vector<string> lineElements = {"fscale", "=", "3"};
    _inputFileReader->parseFScale(lineElements);
    EXPECT_EQ(_engine.getSettings().getFScale(), 3);
    lineElements = {"temperature", "=", "-1"};
    EXPECT_THROW(_inputFileReader->parseFScale(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseNReset)
{
    vector<string> lineElements = {"nreset", "=", "3"};
    _inputFileReader->parseNReset(lineElements);
    EXPECT_EQ(_engine.getSettings().getNReset(), 3);
    lineElements = {"temperature", "=", "-1"};
    EXPECT_THROW(_inputFileReader->parseNReset(lineElements), customException::InputFileException);
}

TEST_F(TestInputFileReader, testParseFReset)
{
    vector<string> lineElements = {"freset", "=", "3"};
    _inputFileReader->parseFReset(lineElements);
    EXPECT_EQ(_engine.getSettings().getFReset(), 3);
    lineElements = {"temperature", "=", "-1"};
    EXPECT_THROW(_inputFileReader->parseFReset(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}