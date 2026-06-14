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
#include "lennardJonesPair.hpp"

namespace
{
    template <class EnergyFunction>
    double centralDifference(
        const EnergyFunction &energyAt,
        const double          distance
    )
    {
        constexpr auto step = 1.0e-5;
        return (energyAt(distance + step) - energyAt(distance - step)) /
               (2.0 * step);
    }
}   // namespace

TEST(TestPairPotentialDerivatives, LennardJonesForceIsNegativeEnergyDerivative)
{
    const auto potential =
        potential::LennardJonesPair(4.0, 0.15, -0.2, -1.0, 1.5);
    const auto distance = 1.7;

    const auto [energy, force] = potential.calculate(distance);
    (void)energy;

    const auto energyDerivative = centralDifference(
        [&potential](const double r) { return potential.calculate(r).first; },
        distance
    );

    EXPECT_NEAR(force, -energyDerivative, 1.0e-7);
}

TEST(TestPairPotentialDerivatives, ShiftedCoulombForceIsNegativeEnergyDerivative)
{
    const auto potential     = potential::CoulombShiftedPotential(4.0);
    const auto distance      = 1.7;
    const auto chargeProduct = 0.75;

    const auto [energy, force] = potential.calculate(distance, chargeProduct);
    (void)energy;

    const auto energyDerivative = centralDifference(
        [&potential, chargeProduct](const double r)
        { return potential.calculate(r, chargeProduct).first; },
        distance
    );

    EXPECT_NEAR(force, -energyDerivative, 1.0e-7);
}
