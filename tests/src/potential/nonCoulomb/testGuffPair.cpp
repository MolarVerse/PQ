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

#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ, InitGo...

#include <cmath>    // for ::pow, ::exp
#include <vector>   // for vector, allocator

#include "gtest/gtest.h"        // for Message, TestPartResult
#include "guffPair.hpp"         // for GuffPair

using namespace potential;

/**
 * @brief tests the calculation of the energy and force of a GuffPair
 *
 */
TEST(TestGuffPair, calculateEnergyAndForces)
{
    const auto coefficients =
        std::vector<double>{2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                            18.0, 19.0, 20.0, 21.0, 22.0, 23.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto guffNonCoulomb =
        potential::GuffPair(rncCutoff, energyCutoff, forceCutoff, coefficients);

    auto distance              = 2.0;
    const auto [energy, force] = guffNonCoulomb.calculate(distance);

    auto energyREF = coefficients[0] / ::pow(distance, coefficients[1]) +
                     coefficients[2] / ::pow(distance, coefficients[3]);
    energyREF += coefficients[4] / ::pow(distance, coefficients[5]) +
                 coefficients[6] / ::pow(distance, coefficients[7]);
    energyREF += coefficients[8] /
                 (1 + ::exp(coefficients[9] * (distance - coefficients[10])));
    energyREF += coefficients[11] /
                 (1 + ::exp(coefficients[12] * (distance - coefficients[13])));
    energyREF += coefficients[14] *
                 ::exp(
                     coefficients[15] *
                     ::pow(distance - coefficients[16], coefficients[17])
                 );
    energyREF += coefficients[18] *
                 ::exp(
                     coefficients[19] *
                     ::pow(distance - coefficients[20], coefficients[21])
                 );
    energyREF += -energyCutoff - forceCutoff * (rncCutoff - distance);

    auto forceREF = coefficients[0] * coefficients[1] /
                    ::pow(distance, coefficients[1] + 1);
    forceREF += coefficients[2] * coefficients[3] /
                ::pow(distance, coefficients[3] + 1);
    forceREF += coefficients[4] * coefficients[5] /
                ::pow(distance, coefficients[5] + 1);
    forceREF += coefficients[6] * coefficients[7] /
                ::pow(distance, coefficients[7] + 1);
    forceREF +=
        coefficients[8] * coefficients[9] *
        ::exp(coefficients[9] * (distance - coefficients[10])) /
        ::pow(1 + ::exp(coefficients[9] * (distance - coefficients[10])), 2);
    forceREF +=
        coefficients[11] * coefficients[12] *
        ::exp(coefficients[12] * (distance - coefficients[13])) /
        ::pow(1 + ::exp(coefficients[12] * (distance - coefficients[13])), 2);
    forceREF -= coefficients[14] * coefficients[15] *
                ::exp(
                    coefficients[15] *
                    ::pow(distance - coefficients[16], coefficients[17])
                ) *
                ::pow(distance - coefficients[16], coefficients[17] - 1);
    forceREF -= coefficients[18] * coefficients[19] *
                ::exp(
                    coefficients[19] *
                    ::pow(distance - coefficients[20], coefficients[21])
                ) *
                ::pow(distance - coefficients[20], coefficients[21] - 1);
    forceREF -= forceCutoff;

    EXPECT_DOUBLE_EQ(energy, energyREF);
    EXPECT_DOUBLE_EQ(force, forceREF);
}

/**
 * @brief regression test for the zero-coefficient gating: terms with a zero
 * leading coefficient must not contribute, and must not call pow/exp on
 * pathological inputs (e.g. pow(distance - rExp, n) when rExp == distance).
 */
TEST(TestGuffPair, calculateWithSparseCoefficients)
{
    const auto distance     = 2.0;
    const auto rncCutoff    = 3.0;
    const auto energyCutoff = 0.0;
    const auto forceCutoff  = 0.0;

    // Only c1 (inverse-power term) and c9 (sigmoid term) are non-zero. The
    // remaining six contribution blocks must be skipped entirely. The c15
    // and c19 blocks would otherwise compute pow(distance - rExp17/21, n)
    // with rExp17 == rExp21 == distance, which is 0^0 = NaN-prone; the gate
    // keeps the result finite.
    auto coefficients = std::vector<double>(22, 0.0);
    coefficients[0]  = 4.0;        // c1
    coefficients[1]  = 6.0;        // n2
    coefficients[8]  = 1.5;        // c9
    coefficients[9]  = 2.0;        // cexp10
    coefficients[10] = 1.0;        // rExp11
    coefficients[16] = distance;   // rExp17  - would make distance - rExp17 == 0
    coefficients[17] = 3.0;        // n18
    coefficients[20] = distance;   // rExp21
    coefficients[21] = 3.0;        // n22

    const auto pair =
        potential::GuffPair(rncCutoff, energyCutoff, forceCutoff, coefficients);

    const auto [energy, force] = pair.calculate(distance);

    // Only the two non-zero terms contribute. Distance cutoff terms are zero
    // because energyCutoff = forceCutoff = 0.
    const auto expectedEnergy =
        coefficients[0] / ::pow(distance, coefficients[1]) +
        coefficients[8] /
            (1 + ::exp(coefficients[9] * (distance - coefficients[10])));

    const auto expectedForce =
        coefficients[1] * coefficients[0] /
            (::pow(distance, coefficients[1]) * distance) +
        coefficients[8] * coefficients[9] *
            ::exp(coefficients[9] * (distance - coefficients[10])) /
            ::pow(
                1 + ::exp(coefficients[9] * (distance - coefficients[10])),
                2
            );

    EXPECT_DOUBLE_EQ(energy, expectedEnergy);
    EXPECT_DOUBLE_EQ(force, expectedForce);
    EXPECT_FALSE(std::isnan(energy));
    EXPECT_FALSE(std::isnan(force));
}