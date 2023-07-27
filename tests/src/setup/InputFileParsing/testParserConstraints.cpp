#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

/**
 * @brief test for parsing if shake is activated
 *
 */
TEST_F(TestInputFileReader, testParseShakeActivated)
{
    vector<string> lineElements = {"shake", "=", "off"};
    _inputFileReader->parseShakeActivated(lineElements);
    EXPECT_FALSE(_engine.getConstraints().isActivated());
    lineElements = {"shake", "=", "on"};
    _inputFileReader->parseShakeActivated(lineElements);
    EXPECT_TRUE(_engine.getConstraints().isActivated());
    lineElements = {"shake", "=", "1"};
    EXPECT_THROW(_inputFileReader->parseShakeActivated(lineElements), customException::InputFileException);
}

/**
 * @brief test for parsing shake tolerance
 *
 */
TEST_F(TestInputFileReader, testParseShakeTolerance)
{
    vector<string> lineElements = {"shake-tolerance", "=", "0.0001"};
    _inputFileReader->parseShakeTolerance(lineElements);
    EXPECT_DOUBLE_EQ(0.0001, _engine.getSettings().getShakeTolerance());
    lineElements = {"shake-tolerance", "=", "-0.0001"};
    EXPECT_THROW(_inputFileReader->parseShakeTolerance(lineElements), customException::InputFileException);
}

/**
 * @brief test for parsing shake iteration
 *
 */
TEST_F(TestInputFileReader, testParseShakeIteration)
{
    vector<string> lineElements = {"shake-iter", "=", "100"};
    _inputFileReader->parseShakeIteration(lineElements);
    EXPECT_EQ(100, _engine.getSettings().getShakeMaxIter());
    lineElements = {"shake-iter", "=", "-100"};
    EXPECT_THROW(_inputFileReader->parseShakeIteration(lineElements), customException::InputFileException);
}

/**
 * @brief test for parsing rattle tolerance
 *
 */
TEST_F(TestInputFileReader, testParseRattleTolerance)
{
    vector<string> lineElements = {"rattle-tolerance", "=", "0.0001"};
    _inputFileReader->parseRattleTolerance(lineElements);
    EXPECT_DOUBLE_EQ(0.0001, _engine.getSettings().getRattleTolerance());
    lineElements = {"rattle-tolerance", "=", "-0.0001"};
    EXPECT_THROW(_inputFileReader->parseRattleTolerance(lineElements), customException::InputFileException);
}

/**
 * @brief test for parsing rattle iteration
 *
 */
TEST_F(TestInputFileReader, testParseRattleIteration)
{
    vector<string> lineElements = {"rattle-iter", "=", "100"};
    _inputFileReader->parseRattleIteration(lineElements);
    EXPECT_EQ(100, _engine.getSettings().getRattleMaxIter());
    lineElements = {"rattle-iter", "=", "-100"};
    EXPECT_THROW(_inputFileReader->parseRattleIteration(lineElements), customException::InputFileException);
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}