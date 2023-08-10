#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "start_file" command
 *
 */
TEST_F(TestInputFileReader, testStartFileName)
{
    InputFileParserGeneral parser(_engine);
    vector<string>         lineElements = {"startFile_name", "=", "start.xyz"};
    parser.parseStartFilename(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getStartFilename(), "start.xyz");
}

/**
 * @brief tests parsing the "moldescriptor_file" command
 *
 */
TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    InputFileParserGeneral parser(_engine);
    vector<string>         lineElements = {"moldescriptorFile_name", "=", "moldescriptor.txt"};
    parser.parseMoldescriptorFilename(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getMoldescriptorFilename(), "moldescriptor.txt");
}

/**
 * @brief tests parsing the "guff_path" command
 *
 */
TEST_F(TestInputFileReader, testGuffPath)
{
    InputFileParserGeneral parser(_engine);
    vector<string>         lineElements = {"guff_path", "=", "guff"};
    EXPECT_THROW_MSG(parser.parseGuffPath(lineElements, 0),
                     customException::InputFileException,
                     R"(The "guff_path" keyword id deprecated. Please use "guffdat_file" instead.)");
}

/**
 * @brief tests parsing the "guffdat_file" command
 *
 */
TEST_F(TestInputFileReader, guffDatFilename)
{
    InputFileParserGeneral parser(_engine);
    vector<string>         lineElements = {"guffdat_file", "=", "guff.dat"};
    parser.parseGuffDatFilename(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getGuffDatFilename(), "guff.dat");
}

/**
 * @brief tests parsing the "jobtype" command
 *
 * @details if the jobtype is not valid it throws inputFileException - possible jobtypes are: mm-md
 *
 */
TEST_F(TestInputFileReader, testJobType)
{
    InputFileParserGeneral parser(_engine);
    vector<string>         lineElements = {"jobtype", "=", "mm-md"};
    parser.parseJobType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getJobtype(), "MMMD");

    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseJobType(lineElements, 0),
                     customException::InputFileException,
                     "Invalid jobtype \"notValid\" at line 0 in input file");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}