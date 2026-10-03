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

#include "engine.hpp"
#include "exceptions.hpp"
#include "potentialSettings.hpp"
#include "simulationBoxInputParser.hpp"
#include "simulationBoxSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests parsing the "density" command
 */
TEST_F(TestInputFileReader, parseDensity)
{
    EXPECT_EQ(settings::SimulationBoxSettings::getDensitySet(), false);
    input::SimulationBoxInputParser parser(_engine->getSharedSimulationBox());
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("density"));
    const auto& parseFunc = funcMap.at("density");

    const std::vector<std::string> lineElements = {"density", "=", "1.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(_engine->getSimulationBox().getDensity(), 1.0);
    EXPECT_EQ(settings::SimulationBoxSettings::getDensitySet(), true);

    _clearParser(parser);

    const std::vector<std::string> lineElements2 = {"density", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements2, 0),
        exc::InputFileException,
        "Invalid value \"-1.0\" for key \"density\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
    );

    _clearParser(parser);

    const std::vector<std::string> zeroDensity = {"density", "=", "0"};
    EXPECT_THROW_MSG(
        parseFunc(zeroDensity, 0),
        exc::InputFileException,
        "Invalid value \"0\" for key \"density\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
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
    input::SimulationBoxInputParser parser(_engine->getSharedSimulationBox());
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rcoulomb"));
    const auto& parseFunc = funcMap.at("rcoulomb");

    const std::vector<std::string> lineElements = {"rcoulomb", "=", "1.0"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(settings::PotentialSettings::getCoulombRadiusCutOff(), 1.0);

    _clearParser(parser);

    const std::vector<std::string> lineElements2 = {"rcoulomb", "=", "-1.0"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements2, 0),
        exc::InputFileException,
        "Invalid value \"-1.0\" for key \"rcoulomb\" at line 0 in input file: "
        "failed validation with message Value must be greater than 0"
    );
}

TEST_F(TestInputFileReader, parseInitVelocities)
{
    input::SimulationBoxInputParser parser(_engine->getSharedSimulationBox());
    const auto                      funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("init_velocities"));
    const auto& parseFunc = funcMap.at("init_velocities");

    const std::vector<std::string> lineElements = {
        "init_velocities",
        "=",
        "true"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::SimulationBoxSettings::getInitializeVelocities(),
        InitVelocities::TRUE
    );

    _clearParser(parser);

    const std::vector<std::string> lineElements2 = {
        "init_velocities",
        "=",
        "false"
    };
    parseFunc(lineElements2, 0);
    EXPECT_EQ(
        settings::SimulationBoxSettings::getInitializeVelocities(),
        InitVelocities::FALSE
    );

    _clearParser(parser);

    const std::vector<std::string> lineElements3 = {
        "init_velocities",
        "=",
        "force"
    };
    parseFunc(lineElements3, 0);
    EXPECT_EQ(
        settings::SimulationBoxSettings::getInitializeVelocities(),
        InitVelocities::FORCE
    );

    _clearParser(parser);

    const std::vector<std::string> lineElements4 = {
        "init_velocities",
        "=",
        "wrongKeyword"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements4, 0),
        exc::InputFileException,
        "Invalid value \"wrongKeyword\" for key \"init_velocities\" at line 0 "
        "in input file. Allowed values: false, true, force"
    );
}
