#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

TEST_F(TestInputFileReader, testStartFileName)
{
    vector<string> lineElements = {"startfilename", "=", "start.xyz"};
    _inputFileReader->parseStartFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getStartFilename(), "start.xyz");
}

TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    vector<string> lineElements = {"moldescriptorfilename", "=", "moldescriptor.txt"};
    _inputFileReader->parseMoldescriptorFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getMoldescriptorFilename(), "moldescriptor.txt");
}

TEST_F(TestInputFileReader, testGuffPath)
{
    vector<string> lineElements = {"guffpath", "=", "guff"};
    _inputFileReader->parseGuffPath(lineElements);
    EXPECT_EQ(_engine.getSettings().getGuffPath(), "guff");
}

TEST_F(TestInputFileReader, testJobType)
{
    vector<string> lineElements = {"jobtype", "=", "mm-md"};
    _inputFileReader->parseJobType(lineElements);
    EXPECT_EQ(_engine.getSettings().getJobtype(), "MMMD");
    lineElements = {"jobtype", "=", "notvalid"};
    EXPECT_THROW(_inputFileReader->parseJobType(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}