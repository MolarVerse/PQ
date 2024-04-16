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

#include "dihedralForceField.hpp"    // for DihedralForceField
#include "dihedralSection.hpp"       // for DihedralSection
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
 * @brief test dihedral section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionDihedral)
{
    std::vector<std::string>             lineElements = {"1", "2", "3", "4", "7"};
    input::topology::DihedralSection dihedralSection;
    dihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getDihedrals().size(), 1);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getMolecules()[0], &(_engine->getSimulationBox().getMolecules()[0]));
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getMolecules()[1], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getMolecules()[2], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getMolecules()[3], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getAtomIndices()[0], 0);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getAtomIndices()[1], 0);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getAtomIndices()[2], 1);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getAtomIndices()[3], 2);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].getType(), 7);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[0].isLinker(), false);

    lineElements = {"1", "2", "3", "4", "7", "*"};
    dihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getDihedrals()[1].isLinker(), true);

    lineElements = {"1", "1", "2", "3", "4"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "3", "4", "7", "#"};
    EXPECT_THROW(dihedralSection.processSection(lineElements, *_engine), customException::TopologyException);
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyDihedral)
{
    input::topology::DihedralSection dihedralSection;
    EXPECT_THROW(dihedralSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(dihedralSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}