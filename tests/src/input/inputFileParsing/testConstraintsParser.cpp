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

#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "constraintSettings.hpp"   // for settings::ConstraintSettings
#include "constraintsInputParser.hpp"
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for InputFileException
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

/**
 * @brief tests parsing the "shake" command
 *
 * @details if the keyword is not "on", "off", "shake" or "mshake", throws
 * inputFileException, otherwise it sets either shake or mshake active.
 *
 */
TEST_F(TestInputFileReader, testParseShakeActivated)
{
    const auto                   &constraints = _engine->getConstraints();
    input::ConstraintsInputParser parser(constraints);
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("shake"));
    const auto &parseFunc = funcMap.at("shake");

    std::vector<std::string> lineElements = {"shake", "=", "off"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(constraints->isActive());
    EXPECT_FALSE(constraints->isShakeActive());
    EXPECT_FALSE(settings::ConstraintSettings::isShakeActivated());

    clearParser(parser);

    lineElements = {"shake", "=", "on"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(constraints->isActive());
    EXPECT_TRUE(constraints->isShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isShakeActivated());

    clearParser(parser);

    settings::ConstraintSettings::deactivateShake();
    constraints->deactivateShake();

    lineElements = {"shake", "=", "shake"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(constraints->isActive());
    EXPECT_TRUE(constraints->isShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isShakeActivated());

    clearParser(parser);

    settings::ConstraintSettings::deactivateShake();
    constraints->deactivateShake();

    lineElements = {"shake", "=", "mshake"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(constraints->isActive());
    EXPECT_TRUE(constraints->isMShakeActive());
    EXPECT_TRUE(constraints->isShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isShakeActivated());
    EXPECT_TRUE(settings::ConstraintSettings::isMShakeActivated());

    clearParser(parser);

    lineElements = {"shake", "=", "1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1\" for key \"shake\" at line 0 in input file. "
        "Allowed values: OFF, ON, SHAKE, MSHAKE"
    );
}

/**
 * @brief tests parsing the "shake-tolerance" command
 *
 * @details if the tolerance is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseShakeTolerance)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("shake_tolerance"));
    const auto &parseFunc = funcMap.at("shake_tolerance");

    std::vector<std::string> lineElements = {"shake-tolerance", "=", "0.0001"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getShakeTolerance(), 0.0001);

    clearParser(parser);

    lineElements = {"shake-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-0.0001\" for key \"shake-tolerance\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    );

    clearParser(parser);

    lineElements = {"shake-tolerance", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"shake-tolerance\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    );
}

/**
 * @brief tests parsing the "shake-iter" command
 *
 * @details if the number of iterations is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseShakeIteration)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("shake_iter"));
    const auto &parseFunc = funcMap.at("shake_iter");

    std::vector<std::string> lineElements = {"shake-iter", "=", "100"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getShakeMaxIter(), 100);

    clearParser(parser);

    lineElements = {"shake-iter", "=", "-100"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100\" for key \"shake-iter\" at line 0 in input "
        "file. Value must be a positive integer"
    );

    clearParser(parser);

    lineElements = {"shake-iter", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"shake-iter\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );
}

/**
 * @brief tests parsing the "rattle-tolerance" command
 *
 * @details if the tolerance is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRattleTolerance)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rattle_tolerance"));
    const auto &parseFunc = funcMap.at("rattle_tolerance");

    std::vector<std::string> lineElements = {"rattle-tolerance", "=", "0.0001"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getRattleTolerance(), 0.0001);

    clearParser(parser);

    lineElements = {"rattle-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-0.0001\" for key \"rattle-tolerance\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    );

    clearParser(parser);

    lineElements = {"rattle-tolerance", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"rattle-tolerance\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    );
}

/**
 * @brief tests parsing the "rattle-iter" command
 *
 * @details if the number of iterations is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseRattleIteration)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rattle_iter"));
    const auto &parseFunc = funcMap.at("rattle_iter");

    std::vector<std::string> lineElements = {"rattle-iter", "=", "100"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getRattleMaxIter(), 100);

    clearParser(parser);

    lineElements = {"rattle-iter", "=", "-100"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100\" for key \"rattle-iter\" at line 0 in input "
        "file. Value must be a positive integer"
    );

    clearParser(parser);

    lineElements = {"rattle-iter", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"rattle-iter\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );
}

/**
 * @brief tests parsing the "mshake-tolerance" command
 *
 * @details if the tolerance is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseMShakeTolerance)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mshake_tolerance"));
    const auto &parseFunc = funcMap.at("mshake_tolerance");

    std::vector<std::string> lineElements = {"mshake-tolerance", "=", "0.01"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getMShakeTolerance(), 0.01);

    clearParser(parser);

    lineElements = {"mshake-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-0.0001\" for key \"mshake-tolerance\" at line 0 in "
        "input file: failed validation with message Value must be greater than "
        "0"
    );

    clearParser(parser);

    lineElements = {"mshake-tolerance", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"mshake-tolerance\" at line 0 in input "
        "file: failed validation with message Value must be greater than 0"
    );
}

/**
 * @brief tests parsing the "mshake-iter" command
 *
 * @details if the number of iterations is negative, throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseMShakeIteration)
{
    input::ConstraintsInputParser parser(_engine->getConstraints());
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mshake_iter"));
    const auto &parseFunc = funcMap.at("mshake_iter");

    std::vector<std::string> lineElements = {"mshake-iter", "=", "73"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getMShakeMaxIter(), 73);

    clearParser(parser);

    lineElements = {"mshake-iter", "=", "-100"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"-100\" for key \"mshake-iter\" at line 0 in input "
        "file. Value must be a positive integer"
    );

    clearParser(parser);

    lineElements = {"mshake-iter", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"mshake-iter\" at line 0 in input file: "
        "failed validation with message Value must be greater than or equal to "
        "1"
    );
}

/**
 * @brief tests parsing the distance_constraints section
 *
 */
TEST_F(TestInputFileReader, testParseDistanceConstraintsActivated)
{
    const auto                   &constraints = _engine->getConstraints();
    input::ConstraintsInputParser parser(constraints);
    const auto                    funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("distance_constraints"));
    const auto &parseFunc = funcMap.at("distance_constraints");

    std::vector<std::string> lineElements = {"distance-constraints", "=", "on"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(constraints->isActive());
    EXPECT_TRUE(constraints->isDistanceConstraintsActive());
    EXPECT_TRUE(settings::ConstraintSettings::isDistanceConstraintsActivated());

    clearParser(parser);

    lineElements = {"distance-constraints", "=", "off"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(constraints->isActive());
    EXPECT_FALSE(constraints->isDistanceConstraintsActive());
    EXPECT_FALSE(
        settings::ConstraintSettings::isDistanceConstraintsActivated()
    );

    clearParser(parser);

    lineElements = {"distance-constraints", "=", "1"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"1\" for key \"distance-constraints\" at line 0 in "
        "input file. Allowed values: on|off|true|false|yes|no"
    );
}
