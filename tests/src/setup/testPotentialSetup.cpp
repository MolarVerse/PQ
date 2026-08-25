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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ

#include <memory>   // for make_shared

#include "coulombReactionField.hpp"      // for CoulombReactionField
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for ParameterFileException
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "gtest/gtest.h"                 // for Message, TestPartResult
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "moleculeType.hpp"              // for MoleculeType
#include "potentialSettings.hpp"         // for PotentialSettings
#include "potentialSetup.hpp"            // for PotentialSetup, setupPotential
#include "strongTypes.hpp"
#include "testSetup.hpp"   // for TestSetup
#include "testUtils.hpp"
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

using namespace setup;
using namespace settings;
using namespace potential;

/**
 * @brief setup the reaction-field Coulomb potential
 */
TEST_F(TestSetup, setupReactionFieldPotential)
{
    PotentialSettings::setCoulombLongRangeType("reaction-field");
    PotentialSetup potentialSetup(*_engine);

    PotentialSettings::setReactionFieldEpsilon(80.0);
    EXPECT_NO_THROW(potentialSetup.setup());
    test::checkType(
        &(_engine->getPotential()->getCoulombPotential()),
        typeid(CoulombReactionField)
    );

    PotentialSettings::setCoulombLongRangeType("shifted");
}

/**
 * @brief setup the coulomb potential
 */
TEST_F(TestSetup, setupCoulombPotential)
{
    PotentialSettings::setCoulombLongRangeType("shifted");
    PotentialSetup potentialSetup(*_engine);
    potentialSetup.setupCoulomb();

    test::checkType(
        &_engine->getPotential()->getCoulombPotential(),
        typeid(CoulombShiftedPotential)
    );

    PotentialSettings::setCoulombLongRangeType("wolf");
    PotentialSetup potentialSetup2(*_engine);
    potentialSetup2.setup();

    test::checkType(
        &_engine->getPotential()->getCoulombPotential(),
        typeid(CoulombWolf)
    );
    const auto &wolfCoulomb = dynamic_cast<CoulombWolf &>(
        _engine->getPotential()->getCoulombPotential()
    );
    EXPECT_EQ(wolfCoulomb.getKappa(), 0.25);
}

/**
 * @brief setup the non coulomb potential
 */
TEST_F(TestSetup, setupNonCoulombPotential)
{
    _engine->getForceField()->activateNonCoulombic();
    _engine->getPotential()->makeNonCoulombPotential(ForceFieldNonCoulomb());
    PotentialSetup potentialSetup(*_engine);
    potentialSetup.setupNonCoulomb();

    test::checkType(
        &_engine->getPotential()->getNonCoulombPotential(),
        typeid(ForceFieldNonCoulomb)
    );

    _engine->getForceField()->deactivateNonCoulombic();
    PotentialSetup potentialSetup2(*_engine);
    potentialSetup2.setupNonCoulomb();

    test::checkType(
        &_engine->getPotential()->getNonCoulombPotential(),
        typeid(GuffNonCoulomb)
    );
}

/**
 * @brief setup the non coulomb pairs for force field non coulomb
 */
TEST_F(TestSetup, setupNonCoulombicPairs)
{
    _engine->getForceField()->activateNonCoulombic();
    _engine->getPotential()->makeNonCoulombPotential(ForceFieldNonCoulomb());
    PotentialSetup potentialSetup(*_engine);

    auto molecule = molsys::MoleculeType(1);
    molecule.addExternalGlobalVDWType(ExtVdwType{0});
    molecule.addExternalGlobalVDWType(ExtVdwType{1});

    _engine->getSimulationBox().addMoleculeType(molecule);

    EXPECT_THROW_MSG(
        potentialSetup.setupNonCoulombicPairs(),
        customException::ParameterFileException,
        "Not all self interacting non coulombics were set in the noncoulombics "
        "section of the parameter file"
    );

    auto nonCoulombPotential = dynamic_cast<ForceFieldNonCoulomb &>(
        _engine->getPotential()->getNonCoulombPotential()
    );

    const auto zero = ExtVdwType(0);
    const auto one  = ExtVdwType(1);

    auto nonCoulombPair1 =
        LennardJonesPair(zero, zero, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});
    auto nonCoulombPair2 =
        LennardJonesPair(one, zero, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});
    auto nonCoulombPair3 =
        LennardJonesPair(zero, one, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});
    auto nonCoulombPair4 =
        LennardJonesPair(one, one, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});

    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<LennardJonesPair>(nonCoulombPair1)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<LennardJonesPair>(nonCoulombPair2)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<LennardJonesPair>(nonCoulombPair3)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<LennardJonesPair>(nonCoulombPair4)
    );

    _engine->getPotential()->makeNonCoulombPotential(nonCoulombPotential);
    PotentialSetup potentialSetup2(*_engine);

    EXPECT_NO_THROW(potentialSetup2.setupNonCoulombicPairs());
}

/**
 * @brief dummy test for setupPotential - all single components are tested
 * individually - should not throw anything
 *
 */
TEST_F(TestSetup, setupPotential)
{
    EXPECT_NO_THROW(setupPotential(*_engine));

    _engine->getForceField()->activateNonCoulombic();
    _engine->getPotential()->makeNonCoulombPotential(ForceFieldNonCoulomb());
    EXPECT_NO_THROW(setupPotential(*_engine));
}

// TEST_F(TestSetup, setupNonCoulombicPairs)
