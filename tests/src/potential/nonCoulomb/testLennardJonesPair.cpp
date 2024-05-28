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

#include <cmath>     // for pow
#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include "lennardJonesPair.hpp"   // for LennardJonesPair
#include "nonCoulombPair.hpp"     // for potential

using namespace potential;

/**
 * @brief tests the equals operator of BuckinghamPair
 *
 */
TEST(TestLennardJonesPair, equalsOperator)
{
    const size_t vdwType1 = 0;
    const size_t vdwType2 = 1;
    const size_t vdwType3 = 2;
    const auto   nonCoulombPair1 =
        LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 3.0);

    const auto nonCoulombPair2 =
        LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 3.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 =
        LennardJonesPair(vdwType2, vdwType1, 1.0, 2.0, 3.0);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 =
        LennardJonesPair(vdwType1, vdwType3, 1.0, 2.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 =
        LennardJonesPair(vdwType1, vdwType2, 2.0, 2.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 =
        LennardJonesPair(vdwType1, vdwType2, 1.0, 3.0, 3.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 =
        LennardJonesPair(vdwType1, vdwType2, 1.0, 2.0, 4.0);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair7);
}

/**
 * @brief tests the calculation of the energy and force of a LennardJonesPair
 *
 */
TEST(TestLennardJonesPair, calculateEnergyAndForces)
{
    const auto   coefficients = std::vector<double>{2.0, 4.0};
    const auto   rncCutoff    = 3.0;
    const double energyCutoff = 1.0;
    const double forceCutoff  = 2.0;

    const auto potential = potential::LennardJonesPair(
        rncCutoff,
        energyCutoff,
        forceCutoff,
        coefficients[0],
        coefficients[1]
    );

    auto distance        = 2.0;
    auto [energy, force] = potential.calculateEnergyAndForce(distance);

    EXPECT_DOUBLE_EQ(
        energy,
        coefficients[0] / ::pow(distance, 6) +
            coefficients[1] / ::pow(distance, 12) - energyCutoff -
            forceCutoff * (rncCutoff - distance)
    );
    EXPECT_DOUBLE_EQ(
        force,
        6 * coefficients[0] / ::pow(distance, 7) +
            12 * coefficients[1] / ::pow(distance, 13) - forceCutoff
    );
}