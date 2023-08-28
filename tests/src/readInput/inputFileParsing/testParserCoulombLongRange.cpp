#include "engine.hpp"                            // for Engine
#include "exceptions.hpp"                        // for InputFileException
#include "inputFileParser.hpp"                   // for readInput
#include "inputFileParserCoulombLongRange.hpp"   // for InputFileParserCoulombLongRange
#include "potentialSettings.hpp"                 // for PotentialSettings
#include "settings.hpp"                          // for Settings
#include "testInputFileReader.hpp"               // for TestInputFileReader
#include "throwWithMessage.hpp"                  // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TestInfo (ptr only)
#include <iosfwd>          // for std
#include <string>          // for string, allocator
#include <vector>          // for vector

using namespace readInput;

/**
 * @brief tests parsing the "long-range" command
 *
 * @details possible options are none or wolf - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseCoulombLongRange)
{
    InputFileParserCoulombLongRange parser(_engine);
    std::vector<std::string>        lineElements = {"long-range", "=", "none"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), "none");

    lineElements = {"long-range", "=", "wolf"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), "wolf");

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
    std::vector<std::string>        lineElements = {"wolf_param", "=", "1.0"};
    parser.parseWolfParameter(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getWolfParameter(), 1.0);

    lineElements = {"wolf_param", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parser.parseWolfParameter(lineElements, 0), customException::InputFileException, "Wolf parameter cannot be negative");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}