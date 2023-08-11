#include "exceptions.hpp"
#include "inputFileParserNonCoulomb.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "noncoulomb" command
 *
 * @details possible options are "none", "lj" and "buck" - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNonCoulombType)
{
    InputFileParserNonCoulomb parser(_engine);
    vector<string>            lineElements = {"noncoulomb", "=", "none"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "none");

    lineElements = {"noncoulomb", "=", "lj"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "lj");

    lineElements = {"noncoulomb", "=", "buck"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "buck");

    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseNonCoulombType(lineElements, 0),
                     customException::InputFileException,
                     "Invalid nonCoulomb type \"notValid\" at line 0 in input file. Possible options are: lj, buck and none");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}