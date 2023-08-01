#include "nonCoulombPotential.hpp"

#include <cmath>
#include <gtest/gtest.h>

TEST(TestGuffLJ, guffLJ)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0, 3.0, 5.0};
    const auto   rncCutoff    = 3.0;
    double       energy       = 0.0;
    double       force        = 0.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto guffLJ = potential::GuffLennardJones();

    auto distance = 2.0;
    guffLJ.calcNonCoulomb(coefficients, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);

    EXPECT_DOUBLE_EQ(energy,
                     2.0 / pow(distance, 6) + 3.0 / pow(distance, 12) - energyCutoff - forceCutoff * (rncCutoff - distance));

    EXPECT_DOUBLE_EQ(force, 12.0 / pow(distance, 7) + 36.0 / pow(distance, 13) - forceCutoff);
    const auto intermediateForce = force;

    distance = 5.0;
    guffLJ.calcNonCoulomb(coefficients, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);
    EXPECT_DOUBLE_EQ(energy,
                     2.0 / pow(distance, 6) + 3.0 / pow(distance, 12) - energyCutoff - forceCutoff * (rncCutoff - distance));
    EXPECT_DOUBLE_EQ(force, +12.0 / pow(distance, 7) + 36.0 / pow(distance, 13) - forceCutoff + intermediateForce);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}