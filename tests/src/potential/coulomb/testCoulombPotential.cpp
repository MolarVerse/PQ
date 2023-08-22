#include "constants.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace potential;

/**
 * @brief tests the general constructor of a coulombPotential
 */
TEST(TestCoulombPotential, constructor)
{
    auto potential = CoulombShiftedPotential(2.0);
    EXPECT_EQ(potential.getCoulombRadiusCutOff(), 2.0);
    EXPECT_EQ(potential.getCoulombEnergyCutOff(), 0.5);
    EXPECT_EQ(potential.getCoulombForceCutOff(), 0.25);
}

/**
 * @brief tests the setCoulombRadiusCutOff function
 */
TEST(TestCoulombPotential, setCoulombRadiusCutOff)
{
    CoulombPotential::setCoulombRadiusCutOff(2.0);

    EXPECT_EQ(CoulombPotential::getCoulombRadiusCutOff(), 2.0);
    EXPECT_EQ(CoulombPotential::getCoulombEnergyCutOff(), 0.5);
    EXPECT_EQ(CoulombPotential::getCoulombForceCutOff(), 0.25);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}