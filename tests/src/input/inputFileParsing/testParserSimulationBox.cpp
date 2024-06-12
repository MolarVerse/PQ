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

#include <gtest/gtest.h>   // for EXPECT_EQ

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "engine.hpp"                         // for Engine
#include "exceptions.hpp"                     // for InputFileException
#include "gtest/gtest.h"                      // for Message, TestPartResult
#include "inputFileParser.hpp"                // for readInput
#include "inputFileParserSimulationBox.hpp"   // for InputFileParserSimulationBox
#include "potentialSettings.hpp"              // for PotentialSettings
#include "simulationBox.hpp"                  // for SimulationBox
#include "simulationBoxSettings.hpp"          // for SimulationBoxSettings
#include "testInputFileReader.hpp"            // for TestInputFileReader
#include "throwWithMessage.hpp"               // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "density" command
 */
TEST_F(TestInputFileReader, parseDensity)
{
    EXPECT_EQ(settings::SimulationBoxSettings::getDensitySet(), false);
    InputFileParserSimulationBox   parser(*_engine);
    const std::vector<std::string> lineElements = {"density", "=", "1.0"};
    parser.parseDensity(lineElements, 0);
    EXPECT_EQ(_engine->getSimulationBox().getDensity(), 1.0);
    EXPECT_EQ(settings::SimulationBoxSettings::getDensitySet(), true);

    const std::vector<std::string> lineElements2 = {"density", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parser.parseDensity(lineElements2, 0),
        customException::InputFileException,
        "Density must be positive - density = -1"
    );
}

/**
 * @brief tests parsing the "rcoulomb" command
 *
 * @details if the rcoulomb is negative it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, parseCoulombRadius)
{
    InputFileParserSimulationBox   parser(*_engine);
    const std::vector<std::string> lineElements = {"rcoulomb", "=", "1.0"};
    parser.parseCoulombRadius(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombRadiusCutOff(), 1.0);

    const std::vector<std::string> lineElements2 = {"rcoulomb", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parser.parseCoulombRadius(lineElements2, 0),
        customException::InputFileException,
        "Coulomb radius cutoff must be positive - \"-1.0\" at line 0 in input "
        "file"
    );
}

TEST_F(TestInputFileReader, parseInitVelocities)
{
    InputFileParserSimulationBox   parser(*_engine);
    const std::vector<std::string> lineElements = {
        "init_velocities",
        "=",
        "true"
    };
    parser.parseInitializeVelocities(lineElements, 0);
    EXPECT_EQ(settings::SimulationBoxSettings::getInitializeVelocities(), true);

    const std::vector<std::string> lineElements2 = {
        "init_velocities",
        "=",
        "false"
    };
    parser.parseInitializeVelocities(lineElements2, 0);
    EXPECT_EQ(
        settings::SimulationBoxSettings::getInitializeVelocities(),
        false
    );

    const std::vector<std::string> lineElements3 = {
        "init_velocities",
        "=",
        "wrongKeyword"
    };
    EXPECT_THROW_MSG(
        parser.parseInitializeVelocities(lineElements3, 0),
        customException::InputFileException,
        "Invalid value for initialize velocities - \"wrongKeyword\" at line 0 "
        "in input file.\n"
        "Possible options are: true, false"
    );
}