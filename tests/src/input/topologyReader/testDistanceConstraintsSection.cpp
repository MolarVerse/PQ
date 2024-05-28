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

/*
 * @brief Test for the DistanceConstraintsSection class
 *
 */
TEST_F(TestTopologySection, processSectionShake)
{
    std::vector<std::string> lineElements = {"1", "2", "1.0", "2.0", "4", "6"};
    input::topology::DistanceConstraintsSection distanceConstraintsSection;
    distanceConstraintsSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getConstraints().getDistanceConstraints().size(), 1);
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getMolecule1(),
        &(_engine->getSimulationBox().getMolecules()[0])
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getMolecule2(),
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getAtomIndex1(),
        0
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getAtomIndex2(),
        0
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getLowerDistance(
        ),
        1.0
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getUpperDistance(
        ),
        2.0
    );
    EXPECT_EQ(
        _engine->getConstraints().getDistanceConstraints()[0].getSpringConstant(
        ),
        4.0
    );
    EXPECT_EQ(
        _engine->getConstraints()
            .getDistanceConstraints()[0]
            .getDSpringConstantDt(),
        6.0
    );

    // not enough elements
    lineElements = {"1", "1", "1.0", "2", "1"};
    EXPECT_THROW(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );

    // same atom indices
    lineElements = {"1", "1", "1.0", "2", "1", "2"};
    EXPECT_THROW(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );

    // lower distance greater than upper distance
    lineElements = {"1", "2", "2.0", "1.0", "1", "2"};
    EXPECT_THROW(
        distanceConstraintsSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyShake)
{
    input::topology::DistanceConstraintsSection distanceConstraintsSection;
    EXPECT_THROW(
        distanceConstraintsSection.endedNormally(false),
        customException::TopologyException
    );
    EXPECT_NO_THROW(distanceConstraintsSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}