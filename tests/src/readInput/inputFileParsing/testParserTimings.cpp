#include "exceptions.hpp"               // for InputFileException
#include "inputFileParser.hpp"          // for readInput
#include "inputFileParserTimings.hpp"   // for InputFileParserTimings
#include "testInputFileReader.hpp"      // for TestInputFileReader
#include "throwWithMessage.hpp"         // for EXPECT_THROW_MSG
#include "timingsSettings.hpp"          // for TimingsSettings

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "timestep" command
 *
 */
TEST_F(TestInputFileReader, testParseTimestep)
{
    InputFileParserTimings parser(*_engine);
    vector<string>         lineElements = {"timestep", "=", "1"};
    parser.parseTimeStep(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getTimeStep(), 1.0);
}

/**
 * @brief tests parsing the "nsteps" command
 *
 * @details if the number of steps is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfSteps)
{
    InputFileParserTimings parser(*_engine);
    vector<string>         lineElements = {"nsteps", "=", "1000"};
    parser.parseNumberOfSteps(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getNumberOfSteps(), 1000);

    lineElements = {"nsteps", "=", "-1"};
    EXPECT_THROW_MSG(
        parser.parseNumberOfSteps(lineElements, 0), customException::InputFileException, "Number of steps cannot be negative");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}