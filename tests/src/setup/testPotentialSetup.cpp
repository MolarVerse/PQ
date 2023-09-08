#include "coulombPotential.hpp"          // for CoulombPotential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "engine.hpp"                    // for Engine
#include "forceFieldClass.hpp"           // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "potential.hpp"                 // for Potential
#include "potentialSettings.hpp"         // for PotentialSettings
#include "potentialSetup.hpp"            // for PotentialSetup, setupPotential
#include "testSetup.hpp"                 // for TestSetup
#include "throwWithMessage.hpp"          // for throwWithMessage

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ
#include <string>          // for allocator, basic_string

using namespace setup;

/**
 * @brief setup the coulomb potential
 */
TEST_F(TestSetup, setupCoulombPotential)
{
    settings::PotentialSettings::setCoulombLongRangeType("none");
    PotentialSetup potentialSetup(_engine);
    potentialSetup.setupCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getCoulombPotential()), typeid(potential::CoulombShiftedPotential));

    settings::PotentialSettings::setCoulombLongRangeType("wolf");
    PotentialSetup potentialSetup2(_engine);
    potentialSetup2.setup();

    EXPECT_EQ(typeid(_engine.getPotential().getCoulombPotential()), typeid(potential::CoulombWolf));
    const auto &wolfCoulomb = dynamic_cast<potential::CoulombWolf &>(_engine.getPotential().getCoulombPotential());
    EXPECT_EQ(wolfCoulomb.getKappa(), 0.25);
}

/**
 * @brief setup the non coulomb potential
 */
TEST_F(TestSetup, setupNonCoulombPotential)
{
    _engine.getForceField().activateNonCoulombic();
    _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    PotentialSetup potentialSetup(_engine);
    potentialSetup.setupNonCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getNonCoulombPotential()), typeid(potential::ForceFieldNonCoulomb));

    _engine.getForceField().deactivateNonCoulombic();
    PotentialSetup potentialSetup2(_engine);
    potentialSetup2.setupNonCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getNonCoulombPotential()), typeid(potential::GuffNonCoulomb));
}

/**
 * @brief setup the non coulomb pairs for force field non coulomb
 */
TEST_F(TestSetup, setupNonCoulombicPairs)
{
    _engine.getForceField().activateNonCoulombic();
    _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    PotentialSetup potentialSetup(_engine);

    auto molecule = simulationBox::Molecule(1);
    molecule.addExternalGlobalVDWType(0);
    molecule.addExternalGlobalVDWType(1);

    _engine.getSimulationBox().addMoleculeType(molecule);

    EXPECT_THROW_MSG(potentialSetup.setupNonCoulombicPairs(),
                     customException::ParameterFileException,
                     "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    auto nonCoulombPotential = dynamic_cast<potential::ForceFieldNonCoulomb &>(_engine.getPotential().getNonCoulombPotential());
    auto nonCoulombPair1     = potential::LennardJonesPair(size_t(0), size_t(0), 10.0, 2.0, 3.0);
    auto nonCoulombPair2     = potential::LennardJonesPair(size_t(1), size_t(0), 10.0, 2.0, 3.0);
    auto nonCoulombPair3     = potential::LennardJonesPair(size_t(0), size_t(1), 10.0, 2.0, 3.0);
    auto nonCoulombPair4     = potential::LennardJonesPair(size_t(1), size_t(1), 10.0, 2.0, 3.0);

    nonCoulombPotential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombPair1));
    nonCoulombPotential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombPair2));
    nonCoulombPotential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombPair3));
    nonCoulombPotential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombPair4));

    _engine.getPotential().makeNonCoulombPotential(nonCoulombPotential);
    PotentialSetup potentialSetup2(_engine);

    EXPECT_NO_THROW(potentialSetup2.setupNonCoulombicPairs());
}

/**
 * @brief dummy test for setupPotential - all single components are tested individually - should not throw anything
 *
 */
TEST_F(TestSetup, setupPotential)
{
    EXPECT_NO_THROW(setupPotential(_engine));

    _engine.getForceField().activateNonCoulombic();
    _engine.getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    EXPECT_NO_THROW(setupPotential(_engine));
}

// TEST_F(TestSetup, setupNonCoulombicPairs)

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}