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

#include <gtest/gtest.h>   // for TestInfo (ptr only)

#include "coulombLongRangeInputParser.hpp"
#include "exceptions.hpp"            // for exc::InputFileException
#include "potentialSettings.hpp"     // for PotentialSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

/**
 * @brief tests parsing the "long-range" command
 *
 * @details possible options are none or wolf - otherwise throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseCoulombLongRange)
{
    using enum CoulombLongRangeType;

    input::CoulombLongRangeInputParser parser;
    const auto                         funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("long_range"));
    const auto& parseFunc = funcMap.at("long_range");

    std::vector<std::string> lineElements = {"long-range", "=", "none"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), SHIFTED);

    _clearParser(parser);

    lineElements = {"long-range", "=", "reaction-field"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getCoulombLongRangeType(),
        REACTION_FIELD
    );

    _clearParser(parser);

    lineElements = {"long-range", "=", "wolf"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombLongRangeType(), WOLF);

    _clearParser(parser);

    lineElements = {"long-range", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"long_range\" at line 0 in input "
        "file. Allowed values: shifted, reaction_field, wolf, none"
    );
}

/**
 * @brief tests parsing the "wolf_param" command
 *
 * @details if negative throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseWolfParameter)
{
    input::CoulombLongRangeInputParser parser;
    const auto                         funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("wolf_param"));
    const auto& parseFunc = funcMap.at("wolf_param");

    std::vector<std::string> lineElements = {"wolf_param", "=", "1.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getWolfParameter(), 1.0);

    _clearParser(parser);

    lineElements = {"wolf_param", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1.0\" for key \"wolf_param\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    );
}

/**
 * @brief tests parsing the "rf_epsilon" command
 *
 */
TEST_F(TestInputFileReader, testParseReactionFieldEpsilon)
{
    input::CoulombLongRangeInputParser parser;
    const auto                         funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rf_epsilon"));
    const auto& parseFunc = funcMap.at("rf_epsilon");

    std::vector<std::string> lineElements = {"rf-epsilon", "=", "1.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getReactionFieldEpsilon(), 1.0);

    _clearParser(parser);

    lineElements = {"rf-epsilon", "=", "0.999999"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0.999999\" for key \"rf_epsilon\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 1"
    );

    _clearParser(parser);

    lineElements = {"rf-epsilon", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1.0\" for key \"rf_epsilon\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 1"
    );
}
