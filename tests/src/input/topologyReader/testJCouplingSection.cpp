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
#include "jCouplingSection.hpp"
#include "testTopologySection.hpp"

using input::topology::JCouplingSection;
using namespace customException;

TEST_F(TestTopologySection, jCouplingSectionKeyword)
{
    JCouplingSection section;
    EXPECT_EQ(section.keyword(), "j_couplings");
}

TEST_F(TestTopologySection, jCouplingSectionEndedNormally)
{
    JCouplingSection section;
    EXPECT_NO_THROW(section.endedNormally(true));
    EXPECT_THROW(section.endedNormally(false), TopologyException);
}

TEST_F(TestTopologySection, jCouplingSectionProcessFiveElements)
{
    JCouplingSection         section;
    // atom1, atom2, atom3, atom4, type
    std::vector<std::string> lineElements = {"1", "2", "3", "4", "9"};
    section.processSection(lineElements, *_engine);

    const auto &jCouplings = _engine->getForceField().getJCouplings();
    ASSERT_EQ(jCouplings.size(), 1u);

    const auto molecules = jCouplings.front().getMolecules();
    ASSERT_EQ(molecules.size(), 4u);

    auto &simBox = _engine->getSimulationBox();
    EXPECT_EQ(molecules[0], &simBox.getMolecule(0));
    EXPECT_EQ(molecules[1], &simBox.getMolecule(1));
    EXPECT_EQ(molecules[2], &simBox.getMolecule(1));
    EXPECT_EQ(molecules[3], &simBox.getMolecule(1));
}

TEST_F(TestTopologySection, jCouplingSectionThrowsOnWrongElementCount)
{
    JCouplingSection         section;
    std::vector<std::string> lineElements = {"1", "2", "3"};
    EXPECT_THROW(
        section.processSection(lineElements, *_engine),
        TopologyException
    );

    lineElements = {"1", "2", "3", "4", "9", "extra"};
    EXPECT_THROW(
        section.processSection(lineElements, *_engine),
        TopologyException
    );
}

TEST_F(TestTopologySection, jCouplingSectionThrowsOnDuplicateAtomIndices)
{
    JCouplingSection         section;
    // atom1 == atom2 — unique check should fire.
    std::vector<std::string> lineElements = {"1", "1", "2", "3", "9"};
    EXPECT_THROW(
        section.processSection(lineElements, *_engine),
        TopologyException
    );
}
