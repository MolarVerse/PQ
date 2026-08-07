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

#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ

#include <memory>   // for allocator

#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_
#include "coulombPotential.hpp"                      // for potential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "gtest/gtest.h"                 // for Message, TestPartResult

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

    EXPECT_DOUBLE_EQ(
        energy,
        chargeProduct * constants::_COULOMB_PREFACTOR_ *
            (1 / distance - energyCutOff - forceCutoff * (rcCutoff - distance))
    );
    EXPECT_DOUBLE_EQ(
        force,
        chargeProduct * constants::_COULOMB_PREFACTOR_ *
            (1 / (distance * distance) - forceCutoff)
    );
}