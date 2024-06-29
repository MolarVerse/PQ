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

#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS

#include <memory>   // for unique_ptr
#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for InputFileException
#include "generalInputParser.hpp"      // for GeneralInputParser
#include "gtest/gtest.h"               // for Message, TestPartResult
#include "inputFileParser.hpp"         // for readInput
#include "mmmdEngine.hpp"              // for MMMDEngine
#include "optEngine.hpp"               // for MMOptEngine
#include "qmmdEngine.hpp"              // for QMMDEngine
#include "qmmmmdEngine.hpp"            // for QMMMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for Settings
#include "testInputFileReader.hpp"     // for TestInputFileReader
#include "throwWithMessage.hpp"        // for EXPECT_THROW_MSG

using namespace input;
using namespace settings;

/**
 * @brief tests parsing the "jobtype" command
 *
 * @details if the jobtype is not valid it throws inputFileException - possible
 * jobtypes are: mm-md
 *
 */
TEST_F(TestInputFileReader, JobType)
{
    GeneralInputParser       parser(*_engine);
    std::vector<std::string> lineElements = {"jobtype", "=", "mm-md"};
    auto                     engine       = std::unique_ptr<engine::Engine>();
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(Settings::getJobtype(), JobType::MM_MD);
    EXPECT_EQ(Settings::isMMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::MMMDEngine));

    lineElements = {"jobtype", "=", "qm-md"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(Settings::getJobtype(), JobType::QM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::QMMDEngine));

    lineElements = {"jobtype", "=", "qm-rpmd"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(Settings::getJobtype(), JobType::RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::RingPolymerQMMDEngine));

    lineElements = {"jobtype", "=", "qmmm-md"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(Settings::getJobtype(), JobType::QMMM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);
    EXPECT_EQ(Settings::isMMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::QMMMMDEngine));

    lineElements = {"jobtype", "=", "mm-opt"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(Settings::getJobtype(), JobType::MM_OPT);
    EXPECT_EQ(Settings::isOptJobType(), true);
    EXPECT_EQ(Settings::isMMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::OptEngine));

    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseJobTypeForEngine(lineElements, 0, engine),
        customException::InputFileException,
        "Invalid jobtype \"notValid\" in input file - possible values are:\n"
        "- mm-opt\n"
        "- mm-md\n"
        "- qm-md\n"
        "- qmmm-md\n"
        "- qm-rpmd\n"
    );

    EXPECT_NO_THROW(parser.parseJobType(lineElements, 0));
}

/**
 * @brief tests parsing the "dim" command
 *
 */
TEST_F(TestInputFileReader, parseDimensionality)
{
    GeneralInputParser       parser(*_engine);
    std::vector<std::string> lineElements = {"dim", "=", "3"};
    parser.parseDimensionality(lineElements, 0);
    EXPECT_EQ(Settings::getDimensionality(), 3);

    lineElements = {"dim", "=", "3D"};
    parser.parseDimensionality(lineElements, 0);
    EXPECT_EQ(Settings::getDimensionality(), 3);

    lineElements = {"dim", "=", "2"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"2\" in input file\n"
        "Possible values are: 3, 3d"
    );

    lineElements = {"dim", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"1\" in input file\n"
        "Possible values are: 3, 3d"
    );

    lineElements = {"dim", "=", "0"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"0\" in input file\n"
        "Possible values are: 3, 3d"
    );
}

/**
 * @brief tests parsing the "floatingPointType" command
 *
 */
TEST_F(TestInputFileReader, parseFloatingPointType)
{
    GeneralInputParser       parser(*_engine);
    std::vector<std::string> lineElements = {"floatingPointType", "=", "float"};
    parser.parseFloatingPointType(lineElements, 0);
    EXPECT_EQ(Settings::getFloatingPointType(), FPType::FLOAT);

    lineElements = {"floatingPointType", "=", "double"};
    parser.parseFloatingPointType(lineElements, 0);
    EXPECT_EQ(Settings::getFloatingPointType(), FPType::DOUBLE);

    lineElements = {"floatingPointType", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseFloatingPointType(lineElements, 0),
        customException::InputFileException,
        "Invalid floating point type \"notValid\" in input file\n"
        "Possible values are: float, double"
    );
}