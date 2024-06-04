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

#include "constraintSettings.hpp"           // for ConstraintSettings
#include "constraints.hpp"                  // for Constraints
#include "engine.hpp"                       // for Engine
#include "exceptions.hpp"                   // for InputFileException
#include "gtest/gtest.h"                    // for Message, TestPartResult
#include "inputFileParser.hpp"              // for readInput
#include "inputFileParserConstraints.hpp"   // for InputFileParserConstraints
#include "testInputFileReader.hpp"          // for TestInputFileReader
#include "throwWithMessage.hpp"             // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "shake" command
 *
 * @details if the keyword is not "on", "off", "shake" or "mshake", throws
 * inputFileException, otherwise it sets either shake or mshake active.
 *
 */
TEST_F(TestInputFileReader, testParseShakeActivated)
{
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string>   lineElements = {"shake", "=", "off"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_FALSE(_engine->getConstraints().isActive());
    EXPECT_FALSE(_engine->getConstraints().isShakeActive());
    EXPECT_FALSE(settings::ConstraintSettings::isShakeActivated());

    lineElements = {"shake", "=", "on"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_TRUE(_engine->getConstraints().isActive());
    EXPECT_TRUE(_engine->getConstraints().isShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isShakeActivated());

    settings::ConstraintSettings::deactivateShake();
    _engine->getConstraints().deactivateShake();

    lineElements = {"shake", "=", "shake"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_TRUE(_engine->getConstraints().isActive());
    EXPECT_TRUE(_engine->getConstraints().isShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isShakeActivated());

    settings::ConstraintSettings::deactivateShake();
    _engine->getConstraints().deactivateShake();

    lineElements = {"shake", "=", "mshake"};
    parser.parseShakeActivated(lineElements, 0);
    EXPECT_TRUE(_engine->getConstraints().isActive());
    EXPECT_TRUE(_engine->getConstraints().isMShakeActive());
    EXPECT_TRUE(settings::ConstraintSettings::isMShakeActivated());

    lineElements = {"shake", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseShakeActivated(lineElements, 0),
        customException::InputFileException,
        "(Invalid shake keyword \"1\" at line 0 in input file\n Possible "
        "keywords are: \"on\", \"off\", \"shake\", \"mshake\")"
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
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string> lineElements = {"shake-tolerance", "=", "0.0001"};
    parser.parseShakeTolerance(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getShakeTolerance(), 0.0001);

    lineElements = {"shake-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parser.parseShakeTolerance(lineElements, 0),
        customException::InputFileException,
        "Shake tolerance must be positive"
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
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string>   lineElements = {"shake-iter", "=", "100"};
    parser.parseShakeIteration(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getShakeMaxIter(), 100);

    lineElements = {"shake-iter", "=", "-100"};
    EXPECT_THROW_MSG(
        parser.parseShakeIteration(lineElements, 0),
        customException::InputFileException,
        "Maximum shake iterations must be positive"
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
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string> lineElements = {"rattle-tolerance", "=", "0.0001"};
    parser.parseRattleTolerance(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getRattleTolerance(), 0.0001);

    lineElements = {"rattle-tolerance", "=", "-0.0001"};
    EXPECT_THROW_MSG(
        parser.parseRattleTolerance(lineElements, 0),
        customException::InputFileException,
        "Rattle tolerance must be positive"
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
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string>   lineElements = {"rattle-iter", "=", "100"};
    parser.parseRattleIteration(lineElements, 0);
    EXPECT_EQ(settings::ConstraintSettings::getRattleMaxIter(), 100);

    lineElements = {"rattle-iter", "=", "-100"};
    EXPECT_THROW_MSG(
        parser.parseRattleIteration(lineElements, 0),
        customException::InputFileException,
        "Maximum rattle iterations must be positive"
    );
}

/**
 * @brief tests parsing the distance_constraints section
 *
 */
TEST_F(TestInputFileReader, testParseDistanceConstraintsActivated)
{
    InputFileParserConstraints parser(*_engine);
    std::vector<std::string> lineElements = {"distance-constraints", "=", "on"};
    parser.parseDistanceConstraintActivated(lineElements, 0);
    EXPECT_TRUE(_engine->getConstraints().isActive());
    EXPECT_TRUE(_engine->getConstraints().isDistanceConstraintsActive());
    EXPECT_TRUE(settings::ConstraintSettings::isDistanceConstraintsActivated());

    lineElements = {"distance-constraints", "=", "off"};
    parser.parseDistanceConstraintActivated(lineElements, 0);
    EXPECT_FALSE(_engine->getConstraints().isActive());
    EXPECT_FALSE(_engine->getConstraints().isDistanceConstraintsActive());
    EXPECT_FALSE(settings::ConstraintSettings::isDistanceConstraintsActivated()
    );

    lineElements = {"distance-constraints", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseDistanceConstraintActivated(lineElements, 0),
        customException::InputFileException,
        R"(Invalid distance-constraints keyword "1" at line 0 in input file\n Possible keywords are "on" and "off")"
    );
}
