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

#include "testTopologyReader.hpp"

#include "constraints.hpp"          // for Constraints
#include "exceptions.hpp"           // for InputFileException, TopologyException
#include "fileSettings.hpp"         // for FileSettings
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "topologyReader.hpp"       // for TopologyReader

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult

using namespace input::topology;

/**
 * @brief tests isNeeded function
 *
 * @return true if shake is enabled
 * @return true if forceField is enabled
 * @return false
 */
TEST_F(TestTopologyReader, isNeeded)
{
    EXPECT_FALSE(isNeeded(*_engine));

    _engine->getConstraints().activate();
    EXPECT_TRUE(isNeeded(*_engine));

    _engine->getConstraints().deactivate();
    settings::ForceFieldSettings::activate();
    EXPECT_TRUE(isNeeded(*_engine));
}

/**
 * @brief tests determineSection function
 *
 */
TEST_F(TestTopologyReader, determineSection)
{
    EXPECT_NO_THROW([[maybe_unused]] const auto dummy = _topologyReader->determineSection({"shake"}));
    EXPECT_THROW([[maybe_unused]] const auto dummy = _topologyReader->determineSection({"unknown"}),
                 customException::TopologyException);
}

/**
 * @brief tests reading a topology file
 */
TEST_F(TestTopologyReader, read)
{
    EXPECT_NO_THROW(_topologyReader->read());

    _engine->getConstraints().activate();
    EXPECT_NO_THROW(_topologyReader->read());

    settings::FileSettings::unsetIsTopologyFileNameSet();
    EXPECT_THROW(_topologyReader->read(), customException::InputFileException);
}

/**
 * @brief tests the readTopologyFile function
 *
 * @note this test does not check any logic, but it is here for completeness
 */
TEST_F(TestTopologyReader, readTopologyFile)
{
    settings::FileSettings::setTopologyFileName("topology.top");
    input::topology::readTopologyFile(*_engine);
}