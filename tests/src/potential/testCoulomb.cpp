#include "coulombPotential.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(TestGuffCoulomb, guffCoulomb)
{
    const auto   coefficient  = 2.0;
    const auto   rncCutoff    = 3.0;
    double       energy       = 0.0;
    double       force        = 0.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto guffCoulomb = potential::GuffCoulomb();

    auto distance = 2.0;
    guffCoulomb.calcCoulomb(coefficient, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);

    EXPECT_DOUBLE_EQ(energy, coefficient / distance - energyCutoff - forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, coefficient / (distance * distance) - forceCutoff);
    const auto intermediateForce = force;

    distance = 5.0;
    guffCoulomb.calcCoulomb(coefficient, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);
    EXPECT_DOUBLE_EQ(energy, coefficient / distance - energyCutoff - forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, coefficient / (distance * distance) - forceCutoff + intermediateForce);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}