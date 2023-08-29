#include "coulombPotential.hpp"          // for CoulombPotential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "engine.hpp"                    // for Engine
#include "forceField.hpp"                // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "potential.hpp"                 // for Potential
#include "potentialSettings.hpp"         // for PotentialSettings
#include "potentialSetup.hpp"            // for PotentialSetup, setupPotential
#include "testSetup.hpp"                 // for TestSetup

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