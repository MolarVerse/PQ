#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseParameterFilename)
{
    vector<string> lineElements = {"parameter_file", "=", ""};
    EXPECT_THROW(_inputFileReader->parseParameterFilename(lineElements), customException::InputFileException);

    lineElements = {"parameter_file", "=", "param.txt"};
    EXPECT_THROW(_inputFileReader->parseParameterFilename(lineElements), customException::InputFileException);

    lineElements = {"parameter_file", "=", "data/parameterFileReader/param.param"};
    _inputFileReader->parseParameterFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getParameterFilename(), "data/parameterFileReader/param.param");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}