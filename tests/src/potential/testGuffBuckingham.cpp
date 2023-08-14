// #include "nonCoulombPotential.hpp"

// #include <cmath>
// #include <gtest/gtest.h>

// TEST(TestGuffBuckingham, guffBuckingham)
// {
//     const auto   coefficients = std::vector<double>{2.0, 4.0, 3.0};
//     const auto   rncCutoff    = 3.0;
//     double       energy       = 0.0;
//     double       force        = 0.0;
//     const double energyCutoff = 1.0;
//     const double forceCutoff  = 2.0;

//     const auto guffBuckingham = potential::GuffBuckingham();

//     auto distance = 2.0;
//     guffBuckingham.calcNonCoulomb(coefficients, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);

//     auto helper = coefficients[0] * exp(coefficients[1] * distance);

//     EXPECT_DOUBLE_EQ(energy, helper + coefficients[2] / pow(distance, 6) - energyCutoff - forceCutoff * (rncCutoff -
//     distance));

//     EXPECT_DOUBLE_EQ(force, -helper * coefficients[1] + 6.0 * coefficients[2] / pow(distance, 7) - forceCutoff);
//     const auto intermediateForce = force;

//     distance = 5.0;
//     helper   = coefficients[0] * exp(coefficients[1] * distance);
//     guffBuckingham.calcNonCoulomb(coefficients, rncCutoff, distance, energy, force, energyCutoff, forceCutoff);
//     EXPECT_DOUBLE_EQ(energy, helper + coefficients[2] / pow(distance, 6) - energyCutoff - forceCutoff * (rncCutoff -
//     distance)); EXPECT_DOUBLE_EQ(force,
//                      -helper * coefficients[1] + 6.0 * coefficients[2] / pow(distance, 7) - forceCutoff + intermediateForce);
// }

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }