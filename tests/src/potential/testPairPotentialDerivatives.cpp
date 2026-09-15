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

#include "buckinghamPair.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "guffPair.hpp"
#include "lennardJonesPair.hpp"
#include "morsePair.hpp"
#include "strongTypes.hpp"

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

    template <class Calculate>
    void expectForceIsNegativeEnergyDerivative(
        const Calculate &calculate,
        const double     distance,
        const double     tolerance
    )
    {
        const auto [energy, force] = calculate(distance);
        (void) energy;

        const auto energyDerivative = centralDifference(
            [&calculate](const double r) { return calculate(r).first; },
            distance
        );

        EXPECT_NEAR(force, -energyDerivative, tolerance);
    }

    std::array<double, defaults::NUM_GUFF_COEFFICIENTS> buildGuffCoefficients()
    {
        return {0.5, 2.0, -0.2,  4.0,   0.1,  6.0, -0.05, 8.0,
                0.4, 1.2, 1.3,   -0.3,  -1.1, 2.5, 0.02,  -0.5,
                0.3, 2.0, -0.01, -0.25, -0.4, 2.0};
    }
}   // namespace

TEST(TestPairPotentialDerivatives, LennardJonesForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::LennardJonesPair(
        4.0,
        0.15,
        -0.2,
        LJParams{.c6 = -1.0, .c12 = 1.5}
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const double r) { return potential.calculate(r); },
        1.7,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, BuckinghamForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::BuckinghamPair(
        4.0,
        0.25,
        -0.1,
        BuckinghamParams{.scaling = 2.0, .dRho = -1.1, .c6 = -0.4}
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const double r) { return potential.calculate(r); },
        1.6,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, MorseForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::MorsePair(
        4.0,
        0.3,
        -0.2,
        MorseParams{
            .dissociationEnergy  = 2.5,
            .wellWidth           = 1.4,
            .equilibriumDistance = 1.1
        }
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const double r) { return potential.calculate(r); },
        1.7,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, GuffForceIsNegativeEnergyDerivative)
{
    const auto potential =
        pot::GuffPair(4.0, 0.4, -0.2, buildGuffCoefficients());

    expectForceIsNegativeEnergyDerivative(
        [&potential](const double r) { return potential.calculate(r); },
        1.8,
        1.0e-6
    );
}

TEST(
    TestPairPotentialDerivatives,
    ShiftedCoulombForceIsNegativeEnergyDerivative
)
{
    const auto potential     = pot::CoulombShiftedPotential(4.0);
    const auto chargeProduct = 0.75;

    expectForceIsNegativeEnergyDerivative(
        [&potential, chargeProduct](const double r)
        { return potential.calculate(r, chargeProduct); },
        1.7,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, WolfCoulombForceIsNegativeEnergyDerivative)
{
    const auto potential     = pot::CoulombWolf(4.0, 0.25);
    const auto chargeProduct = -0.75;

    expectForceIsNegativeEnergyDerivative(
        [&potential, chargeProduct](const double r)
        { return potential.calculate(r, chargeProduct); },
        1.7,
        1.0e-5
    );
}

TEST(TestPairPotentialDerivatives, ShiftedPotentialsAreZeroAtCutoff)
{
    constexpr auto cutoff        = 4.0;
    constexpr auto chargeProduct = 0.75;

    const auto shiftedCoulomb = pot::CoulombShiftedPotential(cutoff);
    const auto wolfCoulomb    = pot::CoulombWolf(cutoff, 0.25);

    const auto [shiftedCoulombEnergy, shiftedCoulombForce] =
        shiftedCoulomb.calculate(cutoff, chargeProduct);
    const auto [wolfCoulombEnergy, wolfCoulombForce] =
        wolfCoulomb.calculate(cutoff, chargeProduct);

    EXPECT_NEAR(shiftedCoulombEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(shiftedCoulombForce, 0.0, 1.0e-12);
    EXPECT_NEAR(wolfCoulombEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(wolfCoulombForce, 0.0, 1.0e-12);
}

TEST(TestPairPotentialDerivatives, NonCoulombShiftedPairsAreZeroAtCutoff)
{
    constexpr auto cutoff = 4.0;

    const auto lennardJonesUnshifted =
        pot::LennardJonesPair(cutoff, LJParams{.c6 = -1.0, .c12 = 1.5});
    const auto buckinghamUnshifted = pot::BuckinghamPair(
        cutoff,
        BuckinghamParams{.scaling = 2.0, .dRho = -1.1, .c6 = -0.4}
    );
    const auto morseUnshifted = pot::MorsePair(
        cutoff,
        MorseParams{
            .dissociationEnergy  = 2.5,
            .wellWidth           = 1.4,
            .equilibriumDistance = 1.1
        }
    );
    const auto guffUnshifted = pot::GuffPair(cutoff, buildGuffCoefficients());

    const auto [ljEnergyCutoff, ljForceCutoff] =
        lennardJonesUnshifted.calculate(cutoff);
    const auto [buckEnergyCutoff, buckForceCutoff] =
        buckinghamUnshifted.calculate(cutoff);
    const auto [morseEnergyCutoff, morseForceCutoff] =
        morseUnshifted.calculate(cutoff);
    const auto [guffEnergyCutoff, guffForceCutoff] =
        guffUnshifted.calculate(cutoff);

    const auto lennardJones = pot::LennardJonesPair(
        cutoff,
        ljEnergyCutoff,
        ljForceCutoff,
        LJParams{.c6 = -1.0, .c12 = 1.5}
    );
    const auto buckingham = pot::BuckinghamPair(
        cutoff,
        buckEnergyCutoff,
        buckForceCutoff,
        BuckinghamParams{.scaling = 2.0, .dRho = -1.1, .c6 = -0.4}
    );
    const auto morse = pot::MorsePair(
        cutoff,
        morseEnergyCutoff,
        morseForceCutoff,
        MorseParams{
            .dissociationEnergy  = 2.5,
            .wellWidth           = 1.4,
            .equilibriumDistance = 1.1
        }
    );
    const auto guff = pot::GuffPair(
        cutoff,
        guffEnergyCutoff,
        guffForceCutoff,
        buildGuffCoefficients()
    );

    const auto [ljEnergy, ljForce]       = lennardJones.calculate(cutoff);
    const auto [buckEnergy, buckForce]   = buckingham.calculate(cutoff);
    const auto [morseEnergy, morseForce] = morse.calculate(cutoff);
    const auto [guffEnergy, guffForce]   = guff.calculate(cutoff);

    EXPECT_NEAR(ljEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(ljForce, 0.0, 1.0e-12);
    EXPECT_NEAR(buckEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(buckForce, 0.0, 1.0e-12);
    EXPECT_NEAR(morseEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(morseForce, 0.0, 1.0e-12);
    EXPECT_NEAR(guffEnergy, 0.0, 1.0e-12);
    EXPECT_NEAR(guffForce, 0.0, 1.0e-12);
}
