#include "engine.hpp"                         // for Engine
#include "exceptions.hpp"                     // for InputFileException
#include "inputFileParser.hpp"                // for readInput
#include "inputFileParserParameterFile.hpp"   // for InputFileParserParameterFile
#include "settings.hpp"                       // for Settings
#include "testInputFileReader.hpp"            // for TestInputFileReader
#include "throwWithMessage.hpp"               // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "parameter_file" command
 *
 * @details if the filename is empty or does not exist it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseParameterFilename)
{
    InputFileParserParameterFile parser(_engine);
    vector<string>               lineElements = {"parameter_file", "=", ""};
    EXPECT_THROW_MSG(parser.parseParameterFilename(lineElements, 0),
                     customException::InputFileException,
                     "Parameter filename cannot be empty");

    lineElements = {"parameter_file", "=", "param.txt"};
    EXPECT_THROW_MSG(parser.parseParameterFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open parameter file - filename = param.txt");

    lineElements = {"parameter_file", "=", "data/parameterFileReader/param.param"};
    parser.parseParameterFilename(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getParameterFilename(), "data/parameterFileReader/param.param");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}