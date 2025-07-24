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

#include "dihedralForceField.hpp"        // for DihedralForceField
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for TopologyException
#include "forceFieldClass.hpp"           // for ForceField
#include "gtest/gtest.h"                 // for Message, TestPartResult
#include "improperDihedralSection.hpp"   // for ImproperDihedralSection
#include "simulationBox.hpp"             // for SimulationBox
#include "testTopologySection.hpp"       // for TestTopologySection

/**
 * @brief test impropers section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionImproperDihedral)
{
    std::vector<std::string> lineElements = {"1", "2", "3", "4", "7"};
    input::topology::ImproperDihedralSection improperDihedralSection;
    improperDihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals().size(), 1);
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getMolecules()[0],
        &(_engine->getSimulationBox().getMolecules()[0])
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getMolecules()[1],
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getMolecules()[2],
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getMolecules()[3],
        &(_engine->getSimulationBox().getMolecules()[1])
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[0],
        0
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[1],
        0
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[2],
        1
    );
    EXPECT_EQ(
        _engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[3],
        2
    );
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getType(), 7);

    lineElements = {"1", "1", "2", "3", "4"};
    EXPECT_THROW(
        improperDihedralSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );

    lineElements = {"1", "2", "7"};
    EXPECT_THROW(
        improperDihedralSection.processSection(lineElements, *_engine),
        customException::TopologyException
    );
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyImproperDihedral)
{
    input::topology::ImproperDihedralSection improperDihedralSection;
    EXPECT_THROW(
        improperDihedralSection.endedNormally(false),
        customException::TopologyException
    );
    EXPECT_NO_THROW(improperDihedralSection.endedNormally(true));
}