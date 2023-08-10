#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "shake" command
 *
 * @details possible options are on or off - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseShakeActivated)
{
    InputFileParserConstraints parser(_engine);
    vector<string>             lineElements = {"shake", "=", "off"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_FALSE(_engine.getConstraints().isActivated());

    lineElements = {"shake", "=", "on"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_TRUE(_engine.getConstraints().isActivated());

    lineElements = {"shake", "=", "1"};
    EXPECT_THROW_MSG(parser.parseShakeActivated(lineElements, 0),
                     customException::InputFileException,
                     R"(Invalid shake keyword "1" at line 0 in input file\n Possible keywords are "on" and "off")");
}

/**
 * @brief tests parsing the "shake-tolerance" command
 *
 * @details if the tolerance is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseShakeTolerance)
{
    InputFileParserConstraints parser(_engine);
    vector<string>             lineElements = {"shake-tolerance", "=", "0.0001"};
    parser.parseShakeTolerance(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getShakeTolerance(), 0.0001);

    lineElements = {"shake-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parser.parseShakeTolerance(lineElements, 0), customException::InputFileException, "Shake tolerance must be positive");
}

/**
 * @brief tests parsing the "shake-iter" command
 *
 * @details if the number of iterations is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseShakeIteration)
{
    InputFileParserConstraints parser(_engine);
    vector<string>             lineElements = {"shake-iter", "=", "100"};
    parser.parseShakeIteration(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getShakeMaxIter(), 100);

    lineElements = {"shake-iter", "=", "-100"};
    EXPECT_THROW_MSG(parser.parseShakeIteration(lineElements, 0),
                     customException::InputFileException,
                     "Maximum shake iterations must be positive");
}

/**
 * @brief tests parsing the "rattle-tolerance" command
 *
 * @details if the tolerance is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRattleTolerance)
{
    InputFileParserConstraints parser(_engine);
    vector<string>             lineElements = {"rattle-tolerance", "=", "0.0001"};
    parser.parseRattleTolerance(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getRattleTolerance(), 0.0001);

    lineElements = {"rattle-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parser.parseRattleTolerance(lineElements, 0), customException::InputFileException, "Rattle tolerance must be positive");
}

/**
 * @brief tests parsing the "rattle-iter" command
 *
 * @details if the number of iterations is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRattleIteration)
{
    InputFileParserConstraints parser(_engine);
    vector<string>             lineElements = {"rattle-iter", "=", "100"};
    parser.parseRattleIteration(lineElements, 0);
    EXPECT_EQ(100, _engine.getSettings().getRattleMaxIter());

    lineElements = {"rattle-iter", "=", "-100"};
    EXPECT_THROW_MSG(parser.parseRattleIteration(lineElements, 0),
                     customException::InputFileException,
                     "Maximum rattle iterations must be positive");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}