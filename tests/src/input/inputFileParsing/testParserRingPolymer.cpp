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

#include "exceptions.hpp"                   // for InputFileException
#include "gtest/gtest.h"                    // for Message, TestPartResult
#include "inputFileParser.hpp"              // for readInput
#include "inputFileParserRingPolymer.hpp"   // for InputFileParserRingPolymer
#include "ringPolymerSettings.hpp"          // for RingPolymerSettings
#include "testInputFileReader.hpp"          // for TestInputFileReader
#include "throwWithMessage.hpp"             // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "rpmd_n_replica" command
 *
 * @details if the number of replicas is lower than 2 it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfReplicas)
{
    InputFileParserRingPolymer parser(*_engine);
    std::vector<std::string>   lineElements = {"rpmd_n_replica", "=", "10"};
    parser.parseNumberOfBeads(lineElements, 0);

    EXPECT_EQ(settings::RingPolymerSettings::getNumberOfBeads(), 10);

    lineElements = {"rpmd_n_replica", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseNumberOfBeads(lineElements, 0),
        customException::InputFileException,
        "Number of beads must be at least 2 - in input file in line 0"
    );
}