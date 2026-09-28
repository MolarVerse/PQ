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

#include <gtest/gtest.h>   // for EXPECT_FALSE, EXPECT_TRUE

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "MMInputParser.hpp"
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for InputFileException, customException
#include "forceFieldSettings.hpp"    // for ForceFieldSettings
#include "potentialSettings.hpp"     // for PotentialSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

/**
 * @brief tests parsing the "force-field" command
 *
 * @details possible options are on, off or bonded - otherwise throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseForceField)
{
    input::MMInputParser parser(
        _engine->getForceField(),
        _engine->getPotential()
    );
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("force_field"));
    const auto& parseFunc = funcMap.at("force_field");

    std::vector<std::string> lineElements = {"force-field", "=", "on"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ForceFieldSettings::isActive());
    EXPECT_TRUE(_engine->getForceField()->isNonCoulombicActivated());

    clearParser(parser);

    lineElements = {"force-field", "=", "off"};
    parseFunc(lineElements, 0);
    EXPECT_FALSE(settings::ForceFieldSettings::isActive());
    EXPECT_FALSE(_engine->getForceField()->isNonCoulombicActivated());

    clearParser(parser);

    lineElements = {"force-field", "=", "bonded"};
    parseFunc(lineElements, 0);
    EXPECT_TRUE(settings::ForceFieldSettings::isActive());
    EXPECT_FALSE(_engine->getForceField()->isNonCoulombicActivated());

    clearParser(parser);

    lineElements = {"forceField", "=", "notValid"};
    ASSERT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"force-field\" at line 0 in input "
        "file. Allowed values: off, on, bonded"
    );
}

/**
 * @brief tests parsing the "noncoulomb" command
 *
 * @details possible options are "none", "lj" and "buck" - otherwise throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseNonCoulombType)
{
    input::MMInputParser parser(
        _engine->getForceField(),
        _engine->getPotential()
    );
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("noncoulomb"));
    const auto& parseNonCoulombFunc = funcMap.at("noncoulomb");

    std::vector<std::string> lineElements = {"noncoulomb", "=", "guff"};
    parseNonCoulombFunc(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        NonCoulombType::GUFF
    );

    clearParser(parser);

    lineElements = {"noncoulomb", "=", "lj"};
    parseNonCoulombFunc(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        NonCoulombType::LJ
    );

    clearParser(parser);

    lineElements = {"noncoulomb", "=", "buck"};
    parseNonCoulombFunc(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        NonCoulombType::BUCKINGHAM
    );

    clearParser(parser);

    lineElements = {"noncoulomb", "=", "morse"};
    parseNonCoulombFunc(lineElements, 0);
    EXPECT_EQ(
        settings::PotentialSettings::getNonCoulombType(),
        NonCoulombType::MORSE
    );

    clearParser(parser);

    lineElements = {"noncoulomb", "=", "notValid"};
    EXPECT_THROW_MSG(
        parseNonCoulombFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"notValid\" for key \"noncoulomb\" at line 0 in input "
        "file. Allowed values: none, lj, buckingham, morse, guff, buck"
    );
}
