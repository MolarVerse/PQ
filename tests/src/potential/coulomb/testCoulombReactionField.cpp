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

#include <gtest/gtest.h>

#include "constants/internalConversionFactors.hpp"
#include "coulombReactionField.hpp"

using namespace potential;

/**
 * @brief tests reaction-field energy, force, and cutoff continuity
 */
TEST(TestCoulombReactionField, calculate)
{
    constexpr auto chargeProduct = 2.0;
    constexpr auto cutoff        = 3.0;
    constexpr auto epsilon       = 80.0;
    constexpr auto distance      = 2.0;

    const auto potential = CoulombReactionField(cutoff, epsilon);

    const auto reactionFieldPrefactor = (epsilon - 1.0) / (2.0 * epsilon + 1.0);
    const auto cutoffEnergy           = 1.0 / cutoff;
    const auto cutoffForce            = 1.0 / (cutoff * cutoff);
    const auto cutoffCubedInverse     = 1.0 / (cutoff * cutoff * cutoff);
    const auto deltaCutoff            = cutoff - distance;
    const auto coulombPrefactor = chargeProduct * constants::COULOMB_PREFACTOR;

    const auto expectedEnergy =
        coulombPrefactor *
        (1.0 / distance - 2.0 * cutoffEnergy + distance * cutoffForce +
         reactionFieldPrefactor * cutoffCubedInverse * deltaCutoff * deltaCutoff
        );
    const auto expectedForce =
        coulombPrefactor *
        (1.0 / (distance * distance) - cutoffForce +
         2.0 * reactionFieldPrefactor * cutoffCubedInverse * deltaCutoff);

    const auto [energy, force] = potential.calculate(distance, chargeProduct);
    EXPECT_DOUBLE_EQ(energy, expectedEnergy);
    EXPECT_DOUBLE_EQ(force, expectedForce);

    const auto [cutoffEnergyValue, cutoffForceValue] =
        potential.calculate(cutoff, chargeProduct);
    EXPECT_NEAR(cutoffEnergyValue, 0.0, 1.0e-12);
    EXPECT_NEAR(cutoffForceValue, 0.0, 1.0e-12);
}
