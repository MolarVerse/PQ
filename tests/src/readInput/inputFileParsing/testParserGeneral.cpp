#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testStartFileName)
{
    vector<string> lineElements = {"startFile_name", "=", "start.xyz"};
    _inputFileReader->parseStartFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getStartFilename(), "start.xyz");
}

TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    vector<string> lineElements = {"moldescriptorFile_name", "=", "moldescriptor.txt"};
    _inputFileReader->parseMoldescriptorFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getMoldescriptorFilename(), "moldescriptor.txt");
}

TEST_F(TestInputFileReader, testGuffPath)
{
    vector<string> lineElements = {"guffpath", "=", "guff"};
    _inputFileReader->parseGuffPath(lineElements);
    EXPECT_EQ(_engine.getSettings().getGuffPath(), "guff");
}

/**
 * @brief tests parsing of guff.dat filename
 *
 */
TEST_F(TestInputFileReader, guffDatFilename)
{
    vector<string> lineElements = {"guffdat_file", "=", "guff.dat"};
    _inputFileReader->parseGuffDatFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getGuffDatFilename(), "guff.dat");
}

TEST_F(TestInputFileReader, testJobType)
{
    vector<string> lineElements = {"jobtype", "=", "mm-md"};
    _inputFileReader->parseJobType(lineElements);
    EXPECT_EQ(_engine.getSettings().getJobtype(), "MMMD");
    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW(_inputFileReader->parseJobType(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}