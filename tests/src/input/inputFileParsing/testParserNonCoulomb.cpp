/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"                  // for InputFileException
#include "gtest/gtest.h"                   // for Message, TestPartResult
#include "inputFileParser.hpp"             // for readInput
#include "inputFileParserNonCoulomb.hpp"   // for InputFileParserNonCoulomb
#include "potentialSettings.hpp"           // for PotentialSettings
#include "testInputFileReader.hpp"         // for TestInputFileReader
#include "throwWithMessage.hpp"            // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "noncoulomb" command
 *
 * @details possible options are "none", "lj" and "buck" - otherwise throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNonCoulombType)
{
    InputFileParserNonCoulomb parser(*_engine);
    std::vector<std::string>  lineElements = {"noncoulomb", "=", "guff"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        settings::NonCoulombType::GUFF
    );

    lineElements = {"noncoulomb", "=", "lj"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        settings::NonCoulombType::LJ
    );

    lineElements = {"noncoulomb", "=", "buck"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        settings::NonCoulombType::BUCKINGHAM
    );

    lineElements = {"noncoulomb", "=", "morse"};
    parser.parseNonCoulombType(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        settings::NonCoulombType::MORSE
    );

    lineElements = {"coulomb", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseNonCoulombType(lineElements, 0),
        customException::InputFileException,
        "Invalid nonCoulomb type \"notValid\" at line 0 in input file.\n"
        "Possible options are: lj, buck, morse and guff"
    );
}