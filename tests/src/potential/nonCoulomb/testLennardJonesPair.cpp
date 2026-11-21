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

#include <cmath>
#include <vector>

#include "lennardJonesPair.hpp"
#include "units.hpp"

/**
 * @brief tests the equals operator of LennardJonesPair
 *
 */
TEST(TestLennardJonesPair, equalsOperator)
{
    const ExtVdwType vdwType1{0};
    const ExtVdwType vdwType2{1};
    const ExtVdwType vdwType3{2};
    const auto       ljParams1 = LJParams{
              .c6  = 2.0 * kcal_per_mol * angstrom6,
              .c12 = 3.0 * kcal_per_mol * angstrom12
    };
    const auto ljParams2 = LJParams{
        .c6  = 3.0 * kcal_per_mol * angstrom6,
        .c12 = 3.0 * kcal_per_mol * angstrom12
    };
    const auto ljParams3 = LJParams{
        .c6  = 2.0 * kcal_per_mol * angstrom6,
        .c12 = 4.0 * kcal_per_mol * angstrom12
    };
    const auto nonCoulombPair1 =
        pot::LennardJonesPair(vdwType1, vdwType2, 1.0 * angstrom, ljParams1);

    const auto nonCoulombPair2 =
        pot::LennardJonesPair(vdwType1, vdwType2, 1.0 * angstrom, ljParams1);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair2);

    const auto nonCoulombPair3 =
        pot::LennardJonesPair(vdwType2, vdwType1, 1.0 * angstrom, ljParams1);
    EXPECT_TRUE(nonCoulombPair1 == nonCoulombPair3);

    const auto nonCoulombPair4 =
        pot::LennardJonesPair(vdwType1, vdwType3, 1.0 * angstrom, ljParams1);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair4);

    const auto nonCoulombPair5 =
        pot::LennardJonesPair(vdwType1, vdwType2, 2.0 * angstrom, ljParams1);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair5);

    const auto nonCoulombPair6 =
        pot::LennardJonesPair(vdwType1, vdwType2, 1.0 * angstrom, ljParams2);
    EXPECT_FALSE(nonCoulombPair1 == nonCoulombPair6);

    const auto nonCoulombPair7 =
        pot::LennardJonesPair(vdwType1, vdwType2, 1.0 * angstrom, ljParams3);
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

    const auto potential = pot::LennardJonesPair(
        rncCutoff * angstrom,
        energyCutoff * kcal_per_mol,
        forceCutoff * kcal_per_mol / angstrom,
        LJParams{
            .c6  = coefficients[0] * kcal_per_mol * angstrom6,
            .c12 = coefficients[1] * kcal_per_mol * angstrom12
        }
    );

    auto distance        = 2.0;
    auto [energy, force] = potential.calculate(distance * angstrom);

    EXPECT_DOUBLE_EQ(
        energy.numerical_value_in(kcal_per_mol),
        (coefficients[0] / ::pow(distance, 6)) +
            (coefficients[1] / ::pow(distance, 12)) - energyCutoff -
            (forceCutoff * (rncCutoff - distance))
    );
    EXPECT_DOUBLE_EQ(
        force.numerical_value_in(kcal_per_mol / angstrom),
        (6 * coefficients[0] / ::pow(distance, 7)) +
            (12 * coefficients[1] / ::pow(distance, 13)) - forceCutoff
    );
}
