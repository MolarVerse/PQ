#include "constants.hpp"          // for _COULOMB_PREFACTOR_
#include "coulombPotential.hpp"   // for potential
#include "coulombWolf.hpp"        // for CoulombWolf

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cmath>           // for erfc, exp, sqrt, M_PI
#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ, Init...
#include <memory>          // for allocator

using namespace potential;

/**
 * @brief tests calculation of Coulomb potential with wolf long-range correction
 *
 */
TEST(TestCoulombWolf, calculate)
{
    const auto   chargeProduct = 2.0;
    const auto   rcCutoff      = 3.0;
    const double kappa         = 0.25;

    const auto guffWolfCoulomb = potential::CoulombWolf(rcCutoff, kappa);
    const auto constParam1     = ::erfc(kappa * rcCutoff) / rcCutoff;
    const auto constParam2     = 2.0 * kappa / ::sqrt(M_PI);
    const auto constParam3     = constParam1 / rcCutoff + constParam2 * ::exp(-kappa * kappa * rcCutoff * rcCutoff) / rcCutoff;

    auto distance = 2.0;

    auto param1                = ::erfc(kappa * distance);
    const auto [energy, force] = guffWolfCoulomb.calculate(distance, chargeProduct);

    EXPECT_DOUBLE_EQ(energy,
                     chargeProduct * constants::_COULOMB_PREFACTOR_ *
                         (param1 / distance - constParam1 + constParam3 * (distance - rcCutoff)));
    EXPECT_DOUBLE_EQ(force,
                     chargeProduct * constants::_COULOMB_PREFACTOR_ *
                         (param1 / (distance * distance) + constParam2 * ::exp(-kappa * kappa * distance * distance) / distance -
                          constParam3));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}