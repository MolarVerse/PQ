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

#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS

#include <vector>   // for vector, allocator

#include "engine.hpp"                    // for Engine
#include "gtest/gtest.h"                 // for Message, TestPartResult
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedSetup.hpp"       // for setupIntraNonBonded
#include "molecule.hpp"                  // for Molecule
#include "testSetup.hpp"                 // for TestSetup

/**
 * @brief tests the setup of the intra non bonded interactions
 *
 */
TEST_F(TestSetup, setupIntraNonBonded)
{
    auto molecule = simulationBox::Molecule(1);
    auto intraNonBondedContainer =
        intraNonBonded::IntraNonBondedContainer(1, {{-1}});

    const auto& intraNonBonded = _engine->getIntraNonBonded();
    intraNonBonded->addIntraNonBondedContainer(intraNonBondedContainer);
    _engine->getSimulationBox().addMolecule(molecule);

    intraNonBonded->deactivate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(intraNonBonded->getIntraNonBondedMaps().size(), 0);

    intraNonBonded->activate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(intraNonBonded->getIntraNonBondedMaps().size(), 1);
}
