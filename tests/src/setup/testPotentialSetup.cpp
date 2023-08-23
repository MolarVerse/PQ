#include "constants.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "guffNonCoulomb.hpp"
#include "potentialSetup.hpp"
#include "testSetup.hpp"

using namespace setup;

/**
 * @brief setup the coulomb potential
 */
TEST_F(TestSetup, setupCoulombPotential)
{
    _engine.getSettings().setCoulombLongRangeType("none");
    PotentialSetup potentialSetup(_engine);
    potentialSetup.setupCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getCoulombPotential()), typeid(potential::CoulombShiftedPotential));

    _engine.getSettings().setCoulombLongRangeType("wolf");
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
    PotentialSetup potentialSetup(_engine);
    potentialSetup.setupNonCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getNonCoulombPotential()), typeid(potential::ForceFieldNonCoulomb));

    _engine.getForceField().deactivateNonCoulombic();
    PotentialSetup potentialSetup2(_engine);
    potentialSetup2.setupNonCoulomb();

    EXPECT_EQ(typeid(_engine.getPotential().getNonCoulombPotential()), typeid(potential::GuffNonCoulomb));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}