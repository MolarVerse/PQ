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

#include "bondConstraint.hpp"        // for BondConstraint
#include "constraints.hpp"           // for Constraints
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for TopologyException
#include "gtest/gtest.h"             // for Message, TestPartResult
#include "shakeSection.hpp"          // for ShakeSection
#include "simulationBox.hpp"         // for SimulationBox
#include "testTopologySection.hpp"   // for TestTopologySection

/**
 * @brief test shake section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionShake)
{
    std::vector<std::string>      lineElements = {"1", "2", "1.0", "0"};
    input::topology::ShakeSection shakeSection;
    shakeSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getConstraints().getBondConstraints().size(), 1);
    EXPECT_EQ(
        _engine->getConstraints().getBondConstraints()[0].getMolecule1(),
        &(_engine->getSimulationBox().getMolecules()[0])
    );
    EXPECT_EQ(
        _engine->getConstraints().getBondConstraints()[0].getMolecule2(),
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        _engine->getConstraints().getBondConstraints()[0].getAtomIndex1(),
        0
    );
    EXPECT_EQ(
        _engine->getConstraints().getBondConstraints()[0].getAtomIndex2(),
        0
    );
    EXPECT_EQ(
        _engine->getConstraints().getBondConstraints()[0].getTargetBondLength(),
        1.0
    );

    lineElements = {"1", "1", "1.0", "0"};
    EXPECT_THROW(
        shakeSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );

    lineElements = {"1", "1", "1.0", "0", "1"};
    EXPECT_THROW(
        shakeSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyShake)
{
    input::topology::ShakeSection shakeSection;
    EXPECT_THROW(
        shakeSection.endedNormally(false),
        customException::TopologyException
    );
    EXPECT_NO_THROW(shakeSection.endedNormally(true));
}