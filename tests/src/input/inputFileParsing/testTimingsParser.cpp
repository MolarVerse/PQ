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

#include <gtest/gtest.h>   // for TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ

#include <iosfwd>   // for std
#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"            // for InputFileException
#include "gtest/gtest.h"             // for Message, TestPartResult, testing
#include "inputFileParser.hpp"       // for readInput
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG
#include "timingsInputParser.hpp"
#include "timingsSettings.hpp"   // for TimingsSettings

using namespace std;
using namespace input;
using namespace ::testing;

/**
 * @brief tests parsing the "timestep" command
 *
 */
TEST_F(TestInputFileReader, testParseTimestep)
{
    TimingsInputParser parser(*_engine);
    vector<string>     lineElements = {"timestep", "=", "1"};
    parser.parseTimeStep(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getTimeStep(), 1.0);
}

/**
 * @brief tests parsing the "nsteps" command
 *
 * @details if the number of steps is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfSteps)
{
    TimingsInputParser parser(*_engine);
    vector<string>     lineElements = {"nsteps", "=", "1000"};
    parser.parseNumberOfSteps(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getNumberOfSteps(), 1000);

    lineElements = {"nsteps", "=", "-1"};
    EXPECT_THROW_MSG(
        parser.parseNumberOfSteps(lineElements, 0),
        customException::InputFileException,
        "Number of steps cannot be negative"
    );
}