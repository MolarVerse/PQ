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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "constraints.hpp"                  // for Constraints
#include "distanceConstraint.hpp"           // for DistanceConstraint
#include "distanceConstraintsSection.hpp"   // for DistanceConstraintsSection
#include "engine.hpp"                       // for Engine
#include "exceptions.hpp"                   // for TopologyException
#include "gtest/gtest.h"                    // for Message, TestPartResult
#include "simulationBox.hpp"                // for SimulationBox
#include "testTopologySection.hpp"          // for TestTopologySection
#include "throwWithMessage.hpp"             // for EXPECT_THROW_MSG

using namespace input::topology;

/*
 * @brief Test for the DistanceConstraintsSection class
 *
 */
TEST_F(TestTopologySection, processSectionDistanceConstraints)
{
    std::vector<std::string> lineElements = {"1", "2", "1.0", "2.0", "4", "6"};
    DistanceConstraintsSection distanceConstraintsSection;
    distanceConstraintsSection.processSection(lineElements, *_engine);

    auto &distanceConstraints = _engine->getConstraints().getDistConstraints();
    auto &molecules           = _engine->getSimulationBox().getMolecules();

    EXPECT_EQ(distanceConstraints.size(), 1);
    EXPECT_EQ(distanceConstraints[0].getMolecule1(), &(molecules[0]));
    EXPECT_EQ(distanceConstraints[0].getMolecule2(), &(molecules[1]));
    EXPECT_EQ(distanceConstraints[0].getAtomIndex1(), 0);
    EXPECT_EQ(distanceConstraints[0].getAtomIndex2(), 0);
    EXPECT_EQ(distanceConstraints[0].getLowerDistance(), 1.0);
    EXPECT_EQ(distanceConstraints[0].getUpperDistance(), 2.0);
    EXPECT_EQ(distanceConstraints[0].getSpringConstant(), 4.0);
    EXPECT_EQ(distanceConstraints[0].getDSpringConstantDt(), 6.0);

    // not enough elements
    lineElements = {"1", "1", "1.0", "2"};
    EXPECT_THROW_MSG(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException,
        "Wrong number of arguments in topology file \"Distance "
        "Constraints\" section at line 0 - number of elements has to be "
        "5 or 6!"
    );

    // same atom indices
    lineElements = {"1", "1", "1.0", "2", "1", "2"};
    EXPECT_THROW_MSG(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException,
        "Topology file \"Distance "
        "Constraints\" at line 0 - atoms cannot be the same!"
    );

    // lower distance greater than upper distance
    lineElements = {"1", "2", "2.0", "1.0", "1", "2"};
    EXPECT_THROW_MSG(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException,
        "Topology file \"Distance "
        "Constraints\" at line 0 - lower distance cannot be greater "
        "than upper distance!"
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyDistanceConstraints)
{
    const DistanceConstraintsSection distanceConstraintsSection{};

    EXPECT_THROW_MSG(
        distanceConstraintsSection.endedNormally(false),
        customException::TopologyException,
        "Topology file error in \"Distance Constraints\" section at line "
        "0 - no end of section found!"
    );

    EXPECT_NO_THROW(distanceConstraintsSection.endedNormally(true));
}