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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "resetKineticsInputParser.hpp"
#include "resetKineticsSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests parsing the "nscale" command
 *
 * @details if the nscale is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseNScale)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("nscale"));
    const auto &parseFunc = funcMap.at("nscale");

    std::vector<std::string> lineElements = {"nscale", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getNScale(), 3);

    _clearParser(parser);

    lineElements = {"nscale", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"nscale\" at line 0 in input file. "
        "Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "fscale" command
 *
 * @details if the fscale is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseFScale)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("fscale"));
    const auto &parseFunc = funcMap.at("fscale");

    std::vector<std::string> lineElements = {"fscale", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getFScale(), 3);

    _clearParser(parser);

    lineElements = {"fscale", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"fscale\" at line 0 in input file. "
        "Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "nreset" command
 *
 * @details if the nreset is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseNReset)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("nreset"));
    const auto &parseFunc = funcMap.at("nreset");

    std::vector<std::string> lineElements = {"nreset", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getNReset(), 3);

    _clearParser(parser);

    lineElements = {"nreset", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"nreset\" at line 0 in input file. "
        "Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "freset" command
 *
 * @details if the freset is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseFReset)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("freset"));
    const auto &parseFunc = funcMap.at("freset");

    std::vector<std::string> lineElements = {"freset", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getFReset(), 3);

    _clearParser(parser);

    lineElements = {"freset", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"freset\" at line 0 in input file. "
        "Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "nreset_angular" command
 *
 * @details if the nreset_angular is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseNResetAngular)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("nreset_angular"));
    const auto &parseFunc = funcMap.at("nreset_angular");

    std::vector<std::string> lineElements = {"nreset_angular", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getNResetAngular(), 3);

    _clearParser(parser);

    lineElements = {"nreset_angular", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"nreset_angular\" at line 0 in input "
        "file. Value must be a positive integer"
    );
}

/**
 * @brief tests parsing the "freset_angular" command
 *
 * @details if the freset_angular is negative it throws inputFileException
 */
TEST_F(TestInputFileReader, testParseFResetAngular)
{
    input::ResetKineticsInputParser parser;
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("freset_angular"));
    const auto &parseFunc = funcMap.at("freset_angular");

    std::vector<std::string> lineElements = {"freset_angular", "=", "3"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ResetKineticsSettings::getFResetAngular(), 3);

    _clearParser(parser);

    lineElements = {"freset_angular", "=", "-1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-1\" for key \"freset_angular\" at line 0 in input "
        "file. Value must be a positive integer"
    );
}
