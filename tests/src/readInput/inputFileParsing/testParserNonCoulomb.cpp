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

TEST_F(TestInputFileReader, parseIntraNonBondedFile)
{
    InputFileParserNonCoulomb parser(_engine);
    vector<string>            lineElements = {"intra-nonBonded_file", "=", "intra.dat"};
    parser.parseIntraNonBondedFile(lineElements, 0);
    EXPECT_EQ(_engine.getSettings().getIntraNonBondedFilename(), "intra.dat");
}

TEST_F(TestInputFileReader, parseIntraNonBondedType)
{
    InputFileParserNonCoulomb parser(_engine);
    vector<string>            lineElements = {"intra-nonBonded_type", "=", "guff"};
    parser.parseIntraNonBondedType(lineElements, 0);
    EXPECT_EQ(_engine.getIntraNonBonded().getIntraNonBondedType(), intraNonBonded::IntraNonBondedType::GUFF);
    EXPECT_TRUE(_engine.getIntraNonBonded().isActivated());

    lineElements = {"intra-nonBonded_type", "=", "force-field"};
    parser.parseIntraNonBondedType(lineElements, 0);
    EXPECT_EQ(_engine.getIntraNonBonded().getIntraNonBondedType(), intraNonBonded::IntraNonBondedType::FORCE_FIELD);
    EXPECT_TRUE(_engine.getIntraNonBonded().isActivated());

    lineElements = {"intra-nonBonded_type", "=", "none"};
    parser.parseIntraNonBondedType(lineElements, 0);
    EXPECT_FALSE(_engine.getIntraNonBonded().isActivated());

    lineElements = {"intra-nonBonded_type", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseIntraNonBondedType(lineElements, 0),
        customException::InputFileException,
        "Invalid intra-nonBonded type \"notValid\" at line 0 in input file. Possible options are: guff, force-field and none");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}