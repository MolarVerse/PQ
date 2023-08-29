#include "engine.hpp"                      // for Engine
#include "exceptions.hpp"                  // for InputFileException
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserNonCoulomb.hpp"   // for InputFileParserNonCoulomb
#include "potentialSettings.hpp"           // for PotentialSettings
#include "settings.hpp"                    // for Settings
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "throwWithMessage.hpp"            // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

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
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), "guff");

    lineElements = {"noncoulomb", "=", "lj"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), "lj");

    lineElements = {"noncoulomb", "=", "buck"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), "buck");

    lineElements = {"noncoulomb", "=", "morse"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), "morse");

    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseNonCoulombType(lineElements, 0),
        customException::InputFileException,
        "Invalid nonCoulomb type \"notValid\" at line 0 in input file. Possible options are: lj, buck, morse and guff");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}