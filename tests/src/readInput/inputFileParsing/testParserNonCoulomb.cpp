#include "exceptions.hpp"
#include "inputFileParserNonCoulomb.hpp"
#include "intraNonBonded.hpp"
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
    vector<string>            lineElements = {"noncoulomb", "=", "guff"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "guff");

    lineElements = {"noncoulomb", "=", "lj"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "lj");

    lineElements = {"noncoulomb", "=", "buck"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "buck");

    lineElements = {"noncoulomb", "=", "morse"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getNonCoulombType(), "morse");

    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseNonCoulombType(lineElements, 0),
        customException::InputFileException,
        "Invalid nonCoulomb type \"notValid\" at line 0 in input file. Possible options are: lj, buck, morse and guff");
}

/**
 * @brief tests parsing the intra non bonded file name
 *
 */
TEST_F(TestInputFileReader, parseIntraNonBondedFile)
{
    InputFileParserNonCoulomb parser(_engine);
    vector<string>            lineElements = {"intra-nonBonded_file", "=", "intra.dat"};
    parser.parseIntraNonBondedFile(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getIntraNonBondedFilename(), "intra.dat");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}