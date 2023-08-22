#include "constants.hpp"
#include "coulombShiftedPotential.hpp"

#include <gtest/gtest.h>

using namespace potential;

/**
 * @brief tests calculation of shifted Coulomb potential
 *
 */
TEST(TestCoulombShiftedPotential, calculate)
{
    const auto chargeProduct = 2.0;
    const auto rcCutoff      = 3.0;
    const auto energyCutOff  = 1 / rcCutoff;
    const auto forceCutoff   = 1 / (rcCutoff * rcCutoff);

    const auto potential = CoulombShiftedPotential(rcCutoff);

    auto distance = 2.0;

    const auto [energy, force] = potential.calculate(distance, chargeProduct);

    EXPECT_DOUBLE_EQ(energy,
                     chargeProduct * constants::_COULOMB_PREFACTOR_ *
                         (1 / distance - energyCutOff - forceCutoff * (rcCutoff - distance)));
    EXPECT_DOUBLE_EQ(force, chargeProduct * constants::_COULOMB_PREFACTOR_ * (1 / (distance * distance) - forceCutoff));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}