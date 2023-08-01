#include "coulombPotential.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(TestGuffCoulomb, guffCoulomb)
{
    const auto   coefficient  = 2.0;
    const auto   rcCutoff     = 3.0;
    double       energy       = 0.0;
    double       force        = 0.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto guffCoulomb = potential::GuffCoulomb(rcCutoff);

    auto distance = 2.0;
    guffCoulomb.calcCoulomb(coefficient, distance, energy, force, energyCutoff, forceCutoff);

    EXPECT_DOUBLE_EQ(energy, coefficient / distance - energyCutoff - forceCutoff * (rcCutoff - distance));
    EXPECT_DOUBLE_EQ(force, coefficient / (distance * distance) - forceCutoff);
    const auto intermediateForce = force;

    distance = 5.0;
    guffCoulomb.calcCoulomb(coefficient, distance, energy, force, energyCutoff, forceCutoff);
    EXPECT_DOUBLE_EQ(energy, coefficient / distance - energyCutoff - forceCutoff * (rcCutoff - distance));
    EXPECT_DOUBLE_EQ(force, coefficient / (distance * distance) - forceCutoff + intermediateForce);
}

TEST(TestGuffCoulomb, guffWolfCoulomb)
{
    const auto   coefficient = 2.0;
    const auto   rcCutoff    = 3.0;
    double       energy      = 0.0;
    double       force       = 0.0;
    const double kappa       = 0.25;

    const auto guffWolfCoulomb = potential::GuffWolfCoulomb(rcCutoff, kappa);
    const auto constParam1     = ::erfc(kappa * rcCutoff) / rcCutoff;
    const auto constParam2     = 2.0 * kappa / ::sqrt(M_PI);
    const auto constParam3     = constParam1 / rcCutoff + constParam2 * ::exp(-kappa * kappa * rcCutoff * rcCutoff) / rcCutoff;

    auto distance = 2.0;

    auto param1 = ::erfc(kappa * distance);
    guffWolfCoulomb.calcCoulomb(coefficient, distance, energy, force, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(energy, coefficient * (param1 / distance - constParam1 + constParam3 * (distance - rcCutoff)));
    EXPECT_DOUBLE_EQ(force,
                     coefficient * (param1 / (distance * distance) +
                                    constParam2 * ::exp(-kappa * kappa * distance * distance) / distance - constParam3));

    distance            = 5.0;
    param1              = ::erfc(kappa * distance);
    const auto oldForce = force;
    guffWolfCoulomb.calcCoulomb(coefficient, distance, energy, force, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(energy, coefficient * (param1 / distance - constParam1 + constParam3 * (distance - rcCutoff)));
    EXPECT_DOUBLE_EQ(force,
                     oldForce +
                         coefficient * (param1 / (distance * distance) +
                                        constParam2 * ::exp(-kappa * kappa * distance * distance) / distance - constParam3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}