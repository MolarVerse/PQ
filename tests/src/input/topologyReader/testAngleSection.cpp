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

#include "angleSection.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "strongTypes.hpp"
#include "testTopologySection.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief test angle section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionAngle)
{
    std::vector<std::string>      lineElements = {"2", "1", "3", "7"};
    input::topology::AngleSection angleSection;
    angleSection.processSection(lineElements, *_engine);

    const auto& angles = _engine->getForceField()->getAngles();

    EXPECT_EQ(angles.size(), 1);
    EXPECT_EQ(
        angles[0].getMolecules()[0],
        _engine->getSimulationBox().getMolecules().data()
    );
    EXPECT_EQ(
        angles[0].getMolecules()[1],
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        angles[0].getMolecules()[2],
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(angles[0].getAtomIndices()[0], AtomIndex{0});
    EXPECT_EQ(angles[0].getAtomIndices()[1], AtomIndex{0});
    EXPECT_EQ(angles[0].getAtomIndices()[2], AtomIndex{1});
    EXPECT_EQ(angles[0].getType(), AngleId{7});
    EXPECT_EQ(angles[0].isLinker(), false);

    lineElements = {"2", "1", "3", "7", "*"};
    angleSection.processSection(lineElements, *_engine);
    EXPECT_EQ(angles[1].isLinker(), true);

    lineElements = {"1", "1", "2", "3"};
    EXPECT_THROW_MSG(
        angleSection.processSection(lineElements, *_engine),
        exc::TopologyException,
        "Topology file angle section at line 0 - atoms cannot be the same!"
    );

    lineElements = {"1", "2", "7"};
    EXPECT_THROW_MSG(
        angleSection.processSection(lineElements, *_engine),
        exc::TopologyException,
        "Wrong number of arguments in topology file angle section at line 0 - "
        "number of elements has to be 4 or 5!"
    );

    lineElements = {"1", "2", "3", "7", "#"};
    EXPECT_THROW_MSG(
        angleSection.processSection(lineElements, *_engine),
        exc::TopologyException,
        "Fifth entry in topology file in angle section has to be a '*' or "
        "empty at line 0!"
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyAngle)
{
    input::topology::AngleSection angleSection;
    EXPECT_THROW_MSG(
        angleSection.endedNormally(false),
        exc::TopologyException,
        "Topology file angle section at line 0 - no end of section found!"
    );
    EXPECT_NO_THROW(angleSection.endedNormally(true));
}
