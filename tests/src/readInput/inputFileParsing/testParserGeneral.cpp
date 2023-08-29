#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for InputFileException
#include "inputFileParser.hpp"          // for readInput
#include "inputFileParserGeneral.hpp"   // for InputFileParserGeneral
#include "settings.hpp"                 // for Settings
#include "testInputFileReader.hpp"      // for TestInputFileReader
#include "throwWithMessage.hpp"         // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace std;
using namespace readInput;
using namespace ::testing;

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