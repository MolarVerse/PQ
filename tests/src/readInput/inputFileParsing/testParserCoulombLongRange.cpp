#include "exceptions.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "long-range" command
 *
 * @details possible options are none or wolf - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseCoulombLongRange)
{
    InputFileParserCoulombLongRange parser(_engine);
    vector<string>                  lineElements = {"long-range", "=", "none"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getCoulombLongRangeType(), "none");

    lineElements = {"long-range", "=", "wolf"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getCoulombLongRangeType(), "wolf");

    lineElements = {"long-range", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseCoulombLongRange(lineElements, 0),
                     customException::InputFileException,
                     "Invalid long-range type for coulomb correction \"notValid\" at line 0 in input file");
}

/**
 * @brief tests parsing the "wolf_param" command
 *
 * @details if negative throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseWolfParameter)
{
    InputFileParserCoulombLongRange parser(_engine);
    vector<string>                  lineElements = {"wolf_param", "=", "1.0"};
    parser.parseWolfParameter(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getWolfParameter(), 1.0);

    lineElements = {"wolf_param", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parser.parseWolfParameter(lineElements, 0), customException::InputFileException, "Wolf parameter cannot be negative");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}