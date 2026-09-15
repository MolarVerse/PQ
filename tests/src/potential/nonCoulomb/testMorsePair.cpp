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

#include <gtest/gtest.h>   // for Test, CmpHelperFloatingPointEQ, EXPECT_EQ

#include <cmath>    // for exp, pow
#include <vector>   // for vector

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include "morsePair.hpp"   // for MorsePair
#include "strongTypes.hpp"

using namespace pot;

/**
 * @brief tests the equals operator of MorsePair
 *
 */
TEST(TestMorsePair, equalsOperator)
{
    const ExtVdwType vdwType1{0};
    const ExtVdwType vdwType2{1};
    const ExtVdwType vdwType3{2};

    const auto nonCoulombPair1 = MorsePair(
        vdwType1,
        vdwType2,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );

    const auto nonCoulombPair2 = MorsePair(
        vdwType1,
        vdwType2,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 = MorsePair(
        vdwType2,
        vdwType1,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 = MorsePair(
        vdwType1,
        vdwType3,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 = MorsePair(
        vdwType1,
        vdwType2,
        2.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 = MorsePair(
        vdwType1,
        vdwType2,
        1.0,
        MorseParams{
            .dissociationEnergy  = 3.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 = MorsePair(
        vdwType1,
        vdwType2,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 4.0,
            .equilibriumDistance = 4.0
        }
    );
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair7);

    const auto nonCoulombPair8 = MorsePair(
        vdwType1,
        vdwType2,
        1.0,
        MorseParams{
            .dissociationEnergy  = 2.0,
            .wellWidth           = 3.0,
            .equilibriumDistance = 5.0
        }
    );
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair8);
}

/**
 * @brief tests the calculation of the energy and force of a MorsePair
 *
 */
TEST(TestMorsePair, calculateEnergyAndForces)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0, 3.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto potential = pot::MorsePair(
        rncCutoff,
        energyCutoff,
        forceCutoff,
        MorseParams{
            .dissociationEnergy  = coefficients[0],
            .wellWidth           = coefficients[1],
            .equilibriumDistance = coefficients[2]
        }
    );

    const auto distance        = 2.0;
    const auto [energy, force] = potential.calculate(distance);

    const auto expTerm = ::exp(-coefficients[1] * (distance - coefficients[2]));

    EXPECT_DOUBLE_EQ(
        energy,
        coefficients[0] * ::pow(1 - expTerm, 2) - energyCutoff -
            forceCutoff * (rncCutoff - distance)
    );
    EXPECT_DOUBLE_EQ(
        force,
        -2 * coefficients[0] * coefficients[1] * (1 - expTerm) * expTerm -
            forceCutoff
    );
}
