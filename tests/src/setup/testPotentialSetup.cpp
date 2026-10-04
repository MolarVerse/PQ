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

#include <memory>

#include "coulombReactionField.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "forceFieldSettings.hpp"
#include "guffNonCoulomb.hpp"
#include "lennardJonesPair.hpp"
#include "moleculeType.hpp"
#include "potentialSettings.hpp"
#include "potentialSetup.hpp"
#include "strongTypes.hpp"
#include "testSetup.hpp"
#include "testUtils.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief setup the reaction-field Coulomb potential
 */
TEST_F(TestSetup, setupReactionFieldPotential)
{
    settings::PotentialSettings::setCoulombLongRangeType(
        CoulombLongRangeType::REACTION_FIELD
    );
    setup::PotentialSetup potentialSetup(*_engine);

    settings::PotentialSettings::setReactionFieldEpsilon(80.0);
    EXPECT_NO_THROW(potentialSetup.setup());
    test::checkType(
        &(_engine->getPotential()->getCoulombPotential()),
        typeid(pot::CoulombReactionField)
    );

    settings::PotentialSettings::setCoulombLongRangeType(
        CoulombLongRangeType::SHIFTED
    );
}

/**
 * @brief setup the coulomb potential
 */
TEST_F(TestSetup, setupCoulombPotential)
{
    settings::PotentialSettings::setCoulombLongRangeType(
        CoulombLongRangeType::SHIFTED
    );
    setup::PotentialSetup potentialSetup(*_engine);
    potentialSetup.setupCoulomb();

    test::checkType(
        &_engine->getPotential()->getCoulombPotential(),
        typeid(pot::CoulombShiftedPotential)
    );

    settings::PotentialSettings::setCoulombLongRangeType(
        CoulombLongRangeType::WOLF
    );
    setup::PotentialSetup potentialSetup2(*_engine);
    potentialSetup2.setup();

    test::checkType(
        &_engine->getPotential()->getCoulombPotential(),
        typeid(pot::CoulombWolf)
    );
    const auto &wolfCoulomb = dynamic_cast<pot::CoulombWolf &>(
        _engine->getPotential()->getCoulombPotential()
    );
    EXPECT_EQ(wolfCoulomb.getKappa(), 0.25);
}

/**
 * @brief setup the non coulomb potential
 */
TEST_F(TestSetup, setupNonCoulombPotential)
{
    settings::ForceFieldSettings::setType(ForceFieldType::ON);
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    setup::PotentialSetup potentialSetup(*_engine);
    potentialSetup.setupNonCoulomb();

    test::checkType(
        &_engine->getPotential()->getNonCoulombPotential(),
        typeid(pot::ForceFieldNonCoulomb)
    );

    settings::ForceFieldSettings::setType(ForceFieldType::OFF);
    setup::PotentialSetup potentialSetup2(*_engine);
    potentialSetup2.setupNonCoulomb();

    test::checkType(
        &_engine->getPotential()->getNonCoulombPotential(),
        typeid(pot::GuffNonCoulomb)
    );
}

/**
 * @brief setup the non coulomb pairs for force field non coulomb
 */
TEST_F(TestSetup, setupNonCoulombicPairs)
{
    settings::ForceFieldSettings::setType(ForceFieldType::ON);
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    setup::PotentialSetup potentialSetup(*_engine);

    auto molecule = molsys::MoleculeType(MolType{1});
    molecule.addExternalGlobalVDWType(ExtVdwType{0});
    molecule.addExternalGlobalVDWType(ExtVdwType{1});

    _engine->getSimulationBox().addMoleculeType(molecule);

    EXPECT_THROW_MSG(
        potentialSetup.setupNonCoulombicPairs(),
        exc::ParameterFileException,
        "Not all self interacting non coulombics were set in the noncoulombics "
        "section of the parameter file"
    );

    auto nonCoulombPotential = dynamic_cast<pot::ForceFieldNonCoulomb &>(
        _engine->getPotential()->getNonCoulombPotential()
    );

    const auto zero = ExtVdwType(0);
    const auto one  = ExtVdwType(1);

    auto nonCoulombPair1 = pot::LennardJonesPair(
        zero,
        zero,
        10.0,
        LJParams{.c6 = 2.0, .c12 = 3.0}
    );
    auto nonCoulombPair2 =
        pot::LennardJonesPair(one, zero, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});
    auto nonCoulombPair3 =
        pot::LennardJonesPair(zero, one, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});
    auto nonCoulombPair4 =
        pot::LennardJonesPair(one, one, 10.0, LJParams{.c6 = 2.0, .c12 = 3.0});

    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(nonCoulombPair1)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(nonCoulombPair2)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(nonCoulombPair3)
    );
    nonCoulombPotential.addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(nonCoulombPair4)
    );

    _engine->getPotential()->makeNonCoulombPotential(nonCoulombPotential);
    setup::PotentialSetup potentialSetup2(*_engine);

    EXPECT_NO_THROW(potentialSetup2.setupNonCoulombicPairs());
}

/**
 * @brief dummy test for setupPotential - all single components are tested
 * individually - should not throw anything
 *
 */
TEST_F(TestSetup, setupPotential)
{
    EXPECT_NO_THROW(setup::setupPotential(*_engine));

    settings::ForceFieldSettings::setType(ForceFieldType::ON);
    _engine->getPotential()->makeNonCoulombPotential(
        pot::ForceFieldNonCoulomb()
    );
    EXPECT_NO_THROW(setup::setupPotential(*_engine));
}
