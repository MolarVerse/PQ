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

#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for InputFileException
#include "gtest/gtest.h"                // for Message, TestPartResult
#include "inputFileParser.hpp"          // for readInput
#include "inputFileParserGeneral.hpp"   // for InputFileParserGeneral
#include "mmmdEngine.hpp"               // for MMMDEngine
#include "qmmdEngine.hpp"               // for QMMDEngine
#include "ringPolymerqmmdEngine.hpp"    // for RingPolymerQMMDEngine
#include "settings.hpp"                 // for Settings
#include "testInputFileReader.hpp"      // for TestInputFileReader
#include "throwWithMessage.hpp"         // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "jobtype" command
 *
 * @details if the jobtype is not valid it throws inputFileException - possible
 * jobtypes are: mm-md
 *
 */
TEST_F(TestInputFileReader, JobType)
{
    InputFileParserGeneral   parser(*_engine);
    std::vector<std::string> lineElements = {"jobtype", "=", "mm-md"};
    auto                     engine       = std::unique_ptr<engine::Engine>();
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(settings::Settings::getJobtype(), settings::JobType::MM_MD);
    EXPECT_EQ(settings::Settings::isMMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::MMMDEngine));

    lineElements = {"jobtype", "=", "qm-md"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(settings::Settings::getJobtype(), settings::JobType::QM_MD);
    EXPECT_EQ(settings::Settings::isQMActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::QMMDEngine));

    lineElements = {"jobtype", "=", "qm-rpmd"};
    parser.parseJobTypeForEngine(lineElements, 0, engine);
    EXPECT_EQ(
        settings::Settings::getJobtype(),
        settings::JobType::RING_POLYMER_QM_MD
    );
    EXPECT_EQ(settings::Settings::isQMActivated(), true);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), true);
    EXPECT_EQ(typeid(*engine), typeid(engine::RingPolymerQMMDEngine));

    lineElements = {"jobtype", "=", "notValid"};
    EXPECT_THROW_MSG(
        parser.parseJobTypeForEngine(lineElements, 0, engine),
        customException::InputFileException,
        "Invalid jobtype \"notValid\" in input file - possible values are: "
        "mm-md, qm-md, qm-rpmd"
    );

    EXPECT_NO_THROW(parser.parseJobType(lineElements, 0));
}

/**
 * @brief tests parsing the "dim" command
 *
 */
TEST_F(TestInputFileReader, parseDimensionality)
{
    InputFileParserGeneral   parser(*_engine);
    std::vector<std::string> lineElements = {"dim", "=", "3"};
    parser.parseDimensionality(lineElements, 0);
    EXPECT_EQ(settings::Settings::getDimensionality(), 3);

    lineElements = {"dim", "=", "3D"};
    parser.parseDimensionality(lineElements, 0);
    EXPECT_EQ(settings::Settings::getDimensionality(), 3);

    lineElements = {"dim", "=", "2"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"2\" in input file - possible values are: 3, "
        "3d"
    );

    lineElements = {"dim", "=", "1"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"1\" in input file - possible values are: 3, "
        "3d"
    );

    lineElements = {"dim", "=", "0"};
    EXPECT_THROW_MSG(
        parser.parseDimensionality(lineElements, 0),
        customException::InputFileException,
        "Invalid dimensionality \"0\" in input file - possible values are: 3, "
        "3d"
    );
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}