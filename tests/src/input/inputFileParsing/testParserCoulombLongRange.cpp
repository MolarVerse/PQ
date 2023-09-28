/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "exceptions.hpp"                        // for InputFileException
#include "inputFileParser.hpp"                   // for readInput
#include "inputFileParserCoulombLongRange.hpp"   // for InputFileParserCoulombLongRange
#include "potentialSettings.hpp"                 // for PotentialSettings
#include "testInputFileReader.hpp"               // for TestInputFileReader
#include "throwWithMessage.hpp"                  // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TestInfo (ptr only)
#include <string>          // for string, allocator
#include <vector>          // for vector

using namespace input;

/**
 * @brief tests parsing the "long-range" command
 *
 * @details possible options are none or wolf - otherwise throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseCoulombLongRange)
{
    InputFileParserCoulombLongRange parser(*_engine);
    std::vector<std::string>        lineElements = {"long-range", "=", "none"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), "none");

    lineElements = {"long-range", "=", "wolf"};
    parser.parseCoulombLongRange(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), "wolf");

    lineElements = {"long-range", "=", "notValid"};
    EXPECT_THROW_MSG(parser.parseCoulombLongRange(lineElements, 0),
                     customException::InputFileException,
                     "Invalid long-range type for coulomb correction \"notValid\" at line 0 in input file - possible options are "
                     "\"none\", \"wolf\"");
}

/**
 * @brief tests parsing the "wolf_param" command
 *
 * @details if negative throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseWolfParameter)
{
    InputFileParserCoulombLongRange parser(*_engine);
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