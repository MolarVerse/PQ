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

#include "coulombShiftedPotential.hpp"
#include "settings.hpp"
#include "units.hpp"

/**
 * @brief tests calculation of shifted Coulomb potential
 *
 */
TEST(TestCoulombShiftedPotential, calculate)
{
    const auto chargeProduct = 2.0 * elementary_charge * elementary_charge;
    const auto rcCutoff      = 3.0 * angstrom;
    const auto energyCutOff  = 1 / rcCutoff;
    const auto forceCutoff   = 1 / (rcCutoff * rcCutoff);

    Settings settings;
    settings.potentialSettings.setCoulombRadiusCutOffByUser(rcCutoff);
    const auto potential =
        pot::CoulombShiftedPotential(settings.potentialSettings);

    auto distance = 2.0 * angstrom;

    const auto [energy, force] = potential.calculate(distance, chargeProduct);

    EXPECT_NEAR(
        energy.numerical_value_in(kcal_per_mol),
        (chargeProduct * coulomb_prefactor *
         (1 / distance - energyCutOff - forceCutoff * (rcCutoff - distance)))
            .numerical_value_in(kcal_per_mol),
        1e-10
    );
    EXPECT_NEAR(
        force.numerical_value_in(kcal_per_mol / angstrom),
        (chargeProduct * coulomb_prefactor *
         (1 / (distance * distance) - forceCutoff))
            .numerical_value_in(kcal_per_mol / angstrom),
        1e-10
    );
}
