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

#include "engine.hpp"                    // for Engine
#include "intraNonBonded.hpp"            // for IntraNonBonded
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedSetup.hpp"       // for setupIntraNonBonded
#include "molecule.hpp"                  // for Molecule
#include "simulationBox.hpp"             // for SimulationBox
#include "testSetup.hpp"                 // for TestSetup

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS
#include <vector>          // for vector, allocator

/**
 * @brief tests the setup of the intra non bonded interactions
 *
 */
TEST_F(TestSetup, setupIntraNonBonded)
{

    auto molecule                = simulationBox::Molecule(1);
    auto intraNonBondedContainer = intraNonBonded::IntraNonBondedContainer(1, {{-1}});

    _engine->getIntraNonBonded().addIntraNonBondedContainer(intraNonBondedContainer);
    _engine->getSimulationBox().addMolecule(molecule);

    _engine->getIntraNonBonded().deactivate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(_engine->getIntraNonBonded().getIntraNonBondedMaps().size(), 0);

    _engine->getIntraNonBonded().activate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(_engine->getIntraNonBonded().getIntraNonBondedMaps().size(), 1);
}