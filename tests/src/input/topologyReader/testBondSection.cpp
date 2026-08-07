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

#include "bondForceField.hpp"        // for BondForceField
#include "bondSection.hpp"           // for BondSection
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for TopologyException
#include "forceFieldClass.hpp"       // for ForceField
#include "simulationBox.hpp"         // for SimulationBox
#include "testTopologySection.hpp"   // for TestTopologySection

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, EXPECT_THROW, TestInfo...
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

/**
 * @brief test bond section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionBond)
{
    std::vector<std::string>         lineElements = {"1", "2", "7"};
    input::topology::BondSection bondSection;
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBonds().size(), 1);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getMolecule1(), &(_engine->getSimulationBox().getMolecules()[0]));
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getMolecule2(), &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getAtomIndex1(), 0);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getAtomIndex2(), 0);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getType(), 7);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].isLinker(), false);

    lineElements = {"1", "2", "7", "*"};
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBonds()[1].isLinker(), true);

    lineElements = {"1", "1", "7"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7", "1", "2"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7", "#"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyBond)
{
    input::topology::BondSection bondSection;
    EXPECT_THROW(bondSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(bondSection.endedNormally(true));
}