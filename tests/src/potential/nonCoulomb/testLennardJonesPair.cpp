#include "lennardJonesPair.hpp"   // for LennardJonesPair
#include "nonCoulombPair.hpp"     // for potential

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <cmath>           // for pow
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ, EXPECT_EQ
#include <string>          // for string
#include <vector>          // for vector

using namespace potential;

/**
 * @brief tests the equals operator of BuckinghamPair
 *
 */
TEST(TestLennardJonesPair, equalsOperator)
{
    const size_t vdwType1        = 0;
    const size_t vdwType2        = 1;
    const size_t vdwType3        = 2;
    const auto   nonCoulombPair1 = LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 3.0);

    const auto nonCoulombPair2 = LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 3.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 = LennardJonesPair(vdwType2, vdwType1, 1.0, 2.0, 3.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 = LennardJonesPair(vdwType1, vdwType3, 1.0, 2.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 = LennardJonesPair(vdwType1, vdwType2, 2.0, 2.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 = LennardJonesPair(vdwType1, vdwType2, 1.0, 3.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 = LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair7);
}

/**
 * @brief tests the calculation of the energy and force of a LennardJonesPair
 *
 */
TEST(TestLennardJonesPair, calculateEnergyAndForces)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto potential = potential::LennardJonesPair(rncCutoff, energyCutoff, forceCutoff, coefficients[0], coefficients[1]);

    auto distance        = 2.0;
    auto [energy, force] = potential.calculateEnergyAndForce(distance);

    EXPECT_DOUBLE_EQ(energy,
                     coefficients[0] / ::pow(distance, 6) + coefficients[1] / ::pow(distance, 12) - energyCutoff -
                         forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, 6 * coefficients[0] / ::pow(distance, 7) + 12 * coefficients[1] / ::pow(distance, 13) - forceCutoff);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}