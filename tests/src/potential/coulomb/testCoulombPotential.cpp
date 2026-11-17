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

#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "units.hpp"

/**
 * @brief tests the general constructor of a coulombPotential
 */
TEST(TestCoulombPotential, constructor)
{
    auto potential = pot::CoulombShiftedPotential(2.0 * angstrom);
    EXPECT_EQ(potential.get_coulombRadiusCutOff(), 2.0 * angstrom);
    EXPECT_EQ(potential.get_coulombEnergyCutOff(), 0.5 / angstrom);
    EXPECT_EQ(potential.get_coulombForceCutOff(), 0.25 / (angstrom * angstrom));
    EXPECT_EQ(
        potential.get_coulombCutOffCubedInverse(),
        1.0 / (2.0 * angstrom * 2.0 * angstrom * 2.0 * angstrom)
    );
}

/**
 * @brief tests the setCoulombRadiusCutOff function
 */
TEST(TestCoulombPotential, setCoulombRadiusCutOff)
{
    pot::CoulombPotential::setCoulombRadiusCutOff(2.0 * angstrom);

    EXPECT_EQ(pot::CoulombPotential::get_coulombRadiusCutOff(), 2.0 * angstrom);
    EXPECT_EQ(
        pot::CoulombPotential::get_coulombCutOffCubedInverse(),
        1.0 / (2.0 * angstrom * 2.0 * angstrom * 2.0 * angstrom)
    );
    EXPECT_EQ(pot::CoulombPotential::get_coulombEnergyCutOff(), 0.5 / angstrom);
    EXPECT_EQ(
        pot::CoulombPotential::get_coulombForceCutOff(),
        0.25 / angstrom2
    );
}
