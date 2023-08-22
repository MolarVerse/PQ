#include "morsePair.hpp"

#include <gtest/gtest.h>

using namespace potential;

/**
 * @brief tests the equals operator of MorsePair
 *
 */
TEST(TestMorsePair, equalsOperator)
{
    const size_t vdwType1        = 0;
    const size_t vdwType2        = 1;
    const size_t vdwType3        = 2;
    const auto   nonCoulombPair1 = MorsePair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 4.0);

    const auto nonCoulombPair2 = MorsePair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 = MorsePair(vdwType2, vdwType1, 1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 = MorsePair(vdwType1, vdwType3, 1.0, 2.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 = MorsePair(vdwType1, vdwType2, 2.0, 2.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 = MorsePair(vdwType1, vdwType2, 1.0, 3.0, 3.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 = MorsePair(vdwType1, vdwType2, 1.0, 2.0, 4.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair7);

    const auto nonCoulombPair8 = MorsePair(vdwType1, vdwType2, 1.0, 2.0, 3.0, 5.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair8);
}

/**
 * @brief tests the calculation of the energy and force of a MorsePair
 *
 */
TEST(TestMorsePair, calculateEnergyAndForces)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0, 3.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto potential =
        potential::MorsePair(rncCutoff, energyCutoff, forceCutoff, coefficients[0], coefficients[1], coefficients[2]);

    const auto distance        = 2.0;
    const auto [energy, force] = potential.calculateEnergyAndForce(distance);

    const auto expTerm = ::exp(-coefficients[1] * (distance - coefficients[2]));

    EXPECT_DOUBLE_EQ(energy, coefficients[0] * ::pow(1 - expTerm, 2) - energyCutoff - forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, -2 * coefficients[0] * coefficients[1] * (1 - expTerm) * expTerm - forceCutoff);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}