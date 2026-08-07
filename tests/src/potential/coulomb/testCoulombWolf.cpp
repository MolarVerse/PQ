/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ, Init...

#include <cmath>    // for erfc, exp, sqrt, M_PI
#include <memory>   // for allocator

#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_
#include "coulombPotential.hpp"                      // for potential
#include "coulombWolf.hpp"                           // for CoulombWolf
#include "gtest/gtest.h"   // for Message, TestPartResult

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
    const auto constParam3 =
        constParam1 / rcCutoff +
        constParam2 * ::exp(-kappa * kappa * rcCutoff * rcCutoff) / rcCutoff;

    auto distance = 2.0;

    auto param1 = ::erfc(kappa * distance);
    const auto [energy, force] =
        guffWolfCoulomb.calculate(distance, chargeProduct);

    EXPECT_DOUBLE_EQ(
        energy,
        chargeProduct * constants::_COULOMB_PREFACTOR_ *
            (param1 / distance - constParam1 +
             constParam3 * (distance - rcCutoff))
    );
    EXPECT_DOUBLE_EQ(
        force,
        chargeProduct * constants::_COULOMB_PREFACTOR_ *
            (param1 / (distance * distance) +
             constParam2 * ::exp(-kappa * kappa * distance * distance) /
                 distance -
             constParam3)
    );
}