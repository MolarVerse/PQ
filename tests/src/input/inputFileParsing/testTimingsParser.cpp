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

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"            // for InputFileException
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
    TimingsInputParser parser;
    const auto         funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("timestep"));
    const auto &timeStepFunc = funcMap.at("timestep");

    vector<string> lineElements = {"timestep", "=", "1"};
    timeStepFunc(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getTimeStep(), 1.0);

    clearParser(parser);

    lineElements = {"timestep", "=", "0"};
    EXPECT_THROW_MSG(
        timeStepFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"timestep\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
    );

    for (const std::string &invalid : {std::string("nan"), std::string("inf")})
    {
        clearParser(parser);
        lineElements = {"timestep", "=", invalid};
        EXPECT_THROW_MSG(
            timeStepFunc(lineElements, 0),
            exc::InputFileException,
            invalid == "nan" ? "Invalid value \"nan\" for key \"timestep\" at "
                               "line 0 in input file: failed validation with "
                               "message Value must not be NaN"
                             : "Invalid value \"inf\" for key \"timestep\" at "
                               "line 0 in input file: failed validation with "
                               "message Value must not be infinite"
        );
    }
}

/**
 * @brief tests parsing the "nsteps" command
 *
 * @details if the number of steps is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNumberOfSteps)
{
    TimingsInputParser parser;
    const auto         funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("nstep"));
    const auto &nstepsFunc = funcMap.at("nstep");

    vector<string> lineElements = {"nstep", "=", "1000"};
    nstepsFunc(lineElements, 0);
    EXPECT_EQ(settings::TimingsSettings::getNumberOfSteps(), 1000);

    clearParser(parser);

    lineElements = {"nstep", "=", "-1"};
    EXPECT_THROW_MSG(
        nstepsFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"nstep\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );

    clearParser(parser);

    lineElements = {"nsteps", "=", "0"};
    EXPECT_THROW_MSG(
        nstepsFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"nstep\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );
}
