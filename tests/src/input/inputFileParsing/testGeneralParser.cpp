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

/**
 * @brief tests parsing the "random_seed" command
 *
 */
TEST_F(TestInputFileReader, parseRandomSeed)
{
    GeneralInputParser       parser(*_engine);
    
    std::vector<std::string> lineElements = {"random_seed", "=", "0"};
    parser.parseRandomSeed(lineElements, 0);
    EXPECT_EQ(Settings::isRandomSeedSet(), true);
    EXPECT_EQ(Settings::getRandomSeed(), 0);
    Settings::setIsRandomSeedSet(false);

    lineElements = {"random_seed", "=", "+73"};
    parser.parseRandomSeed(lineElements, 0);
    EXPECT_EQ(Settings::isRandomSeedSet(), true);
    EXPECT_EQ(Settings::getRandomSeed(), 73);
    Settings::setIsRandomSeedSet(false);

    lineElements = {"random_seed", "=", std::to_string(UINT32_MAX)};
    parser.parseRandomSeed(lineElements, 0);
    EXPECT_EQ(Settings::isRandomSeedSet(), true);
    EXPECT_EQ(Settings::getRandomSeed(), UINT32_MAX);
    Settings::setIsRandomSeedSet(false);

    lineElements = {
        "random_seed",
        "=",
        std::to_string(static_cast<long long>(UINT32_MAX) + 1)
    };
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is out of range.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            static_cast<long long>(UINT32_MAX) + 1,
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "-1"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is out of range.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            -1,
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "seed"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "seed",
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "3.14159"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            3.14159,
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "10e3"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "10e3",
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);

    lineElements = {"random_seed", "=", "+"};
    EXPECT_THROW_MSG(
        parser.parseRandomSeed(lineElements, 0),
        customException::InputFileException,
        std::format(
            "Random seed value \"{}\" is invalid.\n"
            "Must be an integer between \"0\" and \"{}\" (inclusive)",
            "+",
            UINT32_MAX
        )
    );
    EXPECT_EQ(Settings::isRandomSeedSet(), false);
}