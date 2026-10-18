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
#include "shakeSection.hpp"
#include "testTopologySection.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief test shake section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionShake)
{
    std::vector<std::string>      lineElements = {"1", "2", "1.0", "0"};
    input::topology::ShakeSection shakeSection;
    shakeSection.processSection(lineElements, *_engine, 0);

    const auto& constraints = _engine->getConstraints();

    EXPECT_EQ(constraints->getBondConstraints().size(), 1);

    const auto& bondConstraint0 = constraints->getBondConstraints()[0];
    const auto& molecules       = bondConstraint0.molecules();
    const auto& atomIndices     = bondConstraint0.atomIndices();

    EXPECT_EQ(molecules[0], _engine->getSimulationBox().getMolecules().data());
    EXPECT_EQ(molecules[1], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(atomIndices[0], AtomIndex{0});
    EXPECT_EQ(atomIndices[1], AtomIndex{0});
    EXPECT_EQ(bondConstraint0.getTargetBondLength(), 1.0);

    lineElements = {"1", "1", "1.0", "0"};
    EXPECT_THROW_MSG(
        shakeSection.processSection(lineElements, *_engine, 0),
        exc::TopologyException,
        "Topology file shake section at line 0 - atoms cannot be the same!"
    );

    lineElements = {"1", "1", "1.0", "0", "1"};
    EXPECT_THROW_MSG(
        shakeSection.processSection(lineElements, *_engine, 0),
        exc::TopologyException,
        "Wrong number of arguments in topology file shake section at line 0 - "
        "number of elements has to be 3 or 4!"
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyShake)
{
    input::topology::ShakeSection shakeSection;
    EXPECT_THROW_MSG(
        shakeSection.endedNormally(false, 0),
        exc::TopologyException,
        "Topology file shake section at line 0 - no end of section found!"
    );
    EXPECT_NO_THROW(shakeSection.endedNormally(true, 0));
}
