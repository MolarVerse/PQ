#include "buckinghamPair.hpp"

#include <cmath>   // for exp, pow
#include <gtest/gtest.h>

using namespace potential;

/**
 * @brief tests the equals operator of BuckinghamPair
 *
 */
TEST(TestBuckinghamPair, equalsOperator)
{
    const size_t vdwType1        = 0;
    const size_t vdwType2        = 1;
    const size_t vdwType3        = 2;
    const auto   nonCoulombPair1 = BuckinghamPair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 4.0);

    const auto nonCoulombPair2 = BuckinghamPair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 = BuckinghamPair(vdwType2, vdwType1, 1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 = BuckinghamPair(vdwType1, vdwType3, 1.0, 2.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 = BuckinghamPair(vdwType1, vdwType2, 2.0, 2.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 = BuckinghamPair(vdwType1, vdwType2, 1.0, 3.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 = BuckinghamPair(vdwType1, vdwType2, 1.0, 2.0, 4.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair7);

    const auto nonCoulombPair8 = BuckinghamPair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 5.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair8);
}

/**
 * @brief tests the calculation of the energy and force of a BuckinghamPair
 *
 */
TEST(TestBuckinghamPair, calculateEnergyAndForces)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0, 3.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto potential =
        potential::BuckinghamPair(rncCutoff, energyCutoff, forceCutoff, coefficients[0], coefficients[1], coefficients[2]);

    auto distance        = 2.0;
    auto [energy, force] = potential.calculateEnergyAndForce(distance);

    auto helper = coefficients[0] * exp(coefficients[1] * distance);

    EXPECT_DOUBLE_EQ(energy, helper + coefficients[2] / pow(distance, 6) - energyCutoff - forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, -helper * coefficients[1] + 6.0 * coefficients[2] / pow(distance, 7) - forceCutoff);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}