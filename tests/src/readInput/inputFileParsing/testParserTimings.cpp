#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "timestep" command
 *
 */
TEST_F(TestInputFileReader, testParseTimestep)
{
    InputFileParserTimings parser(_engine);
    vector<string>         lineElements = {"timestep", "=", "1"};
    parser.parseTimeStep(lineElements, 0);
    EXPECT_EQ(_engine.getTimings().getTimestep(), 1.0);
}

/**
 * @brief tests parsing the "nsteps" command
 *
 * @details if the number of steps is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfSteps)
{
    InputFileParserTimings parser(_engine);
    vector<string>         lineElements = {"nsteps", "=", "1000"};
    parser.parseNumberOfSteps(lineElements, 0);
    EXPECT_EQ(_engine.getTimings().getNumberOfSteps(), 1000);

    lineElements = {"nsteps", "=", "-1"};
    EXPECT_THROW_MSG(
        parser.parseNumberOfSteps(lineElements, 0), customException::InputFileException, "Number of steps cannot be negative");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}