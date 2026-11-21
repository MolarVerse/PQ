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
#include "potentialSettings.hpp"
#include "strongTypes.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace
{
    template <typename EnergyFunction>
    Force centralDifference(
        const EnergyFunction &energyAt,
        const Distance       &distance
    )
    {
        constexpr auto step = 1.0e-5 * angstrom;
        return (energyAt(distance + step) - energyAt(distance - step)) /
               (2.0 * step);
    }

    template <typename Calculate>
    void expectForceIsNegativeEnergyDerivative(
        const Calculate &calculate,
        const Distance  &distance,
        const double     tolerance
    )
    {
        const auto [energy, force] = calculate(distance);
        (void) energy;

        const auto energyDerivative = centralDifference(
            [&calculate](const Distance &radius)
            { return calculate(radius).first; },
            distance
        );

        EXPECT_NEAR(
            force.numerical_value_in(kcal_per_mol / angstrom),
            -energyDerivative.numerical_value_in(kcal_per_mol / angstrom),
            tolerance
        );
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
        4.0 * angstrom,
        0.15 * kcal_per_mol,
        -0.2 * kcal_per_mol / angstrom,
        LJParams{
            .c6  = -1.0 * kcal_per_mol * angstrom6,
            .c12 = 1.5 * kcal_per_mol * angstrom12
        }
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const Distance &radius)
        { return potential.calculate(radius); },
        1.7 * angstrom,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, BuckinghamForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::BuckinghamPair(
        4.0 * angstrom,
        0.25 * kcal_per_mol,
        -0.1 * kcal_per_mol / angstrom,
        BuckinghamParams{
            .scaling = 2.0 * kcal_per_mol,
            .dRho    = -1.1 / angstrom,
            .c6      = -0.4 * kcal_per_mol * angstrom6
        }
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const Distance &radius)
        { return potential.calculate(radius); },
        1.6 * angstrom,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, MorseForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::MorsePair(
        4.0 * angstrom,
        0.3 * kcal_per_mol,
        -0.2 * kcal_per_mol / angstrom,
        MorseParams{
            .dissociationEnergy  = 2.5 * kcal_per_mol,
            .wellWidth           = 1.4 / angstrom,
            .equilibriumDistance = 1.1 * angstrom
        }
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const Distance &radius)
        { return potential.calculate(radius); },
        1.7 * angstrom,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, GuffForceIsNegativeEnergyDerivative)
{
    const auto potential = pot::GuffPair(
        4.0 * angstrom,
        0.4 * kcal_per_mol,
        -0.2 * kcal_per_mol / angstrom,
        buildGuffCoefficients()
    );

    expectForceIsNegativeEnergyDerivative(
        [&potential](const Distance &radius)
        { return potential.calculate(radius); },
        1.8 * angstrom,
        1.0e-6
    );
}

TEST(
    TestPairPotentialDerivatives,
    ShiftedCoulombForceIsNegativeEnergyDerivative
)
{
    PotentialSettings potentialSettings;
    potentialSettings.setCoulombRadiusCutoffByUser(4.0 * angstrom);
    const auto potential     = pot::CoulombShiftedPotential(potentialSettings);
    const auto chargeProduct = 0.75 * elementary_charge * elementary_charge;

    expectForceIsNegativeEnergyDerivative(
        [&potential, chargeProduct](const Distance &radius)
        { return potential.calculate(radius, chargeProduct); },
        1.7 * angstrom,
        1.0e-7
    );
}

TEST(TestPairPotentialDerivatives, WolfCoulombForceIsNegativeEnergyDerivative)
{
    PotentialSettings potentialSettings;
    potentialSettings.setCoulombRadiusCutoffByUser(4.0 * angstrom);
    const auto potential = pot::CoulombWolf(potentialSettings, 0.25 / angstrom);
    const auto chargeProduct = -0.75 * elementary_charge * elementary_charge;

    expectForceIsNegativeEnergyDerivative(
        [&potential, chargeProduct](const Distance &radius)
        { return potential.calculate(radius, chargeProduct); },
        1.7 * angstrom,
        1.0e-5
    );
}

TEST(TestPairPotentialDerivatives, ShiftedPotentialsAreZeroAtCutoff)
{
    constexpr auto cutoff        = 4.0 * angstrom;
    constexpr auto chargeProduct = 0.75 * elementary_charge * elementary_charge;

    PotentialSettings potentialSettings;
    potentialSettings.setCoulombRadiusCutoffByUser(cutoff);

    const auto shiftedCoulomb = pot::CoulombShiftedPotential(potentialSettings);
    const auto wolfCoulomb =
        pot::CoulombWolf(potentialSettings, 0.25 / angstrom);

    const auto [shiftedCoulombEnergy, shiftedCoulombForce] =
        shiftedCoulomb.calculate(cutoff, chargeProduct);
    const auto [wolfCoulombEnergy, wolfCoulombForce] =
        wolfCoulomb.calculate(cutoff, chargeProduct);

    EXPECT_NEAR(
        shiftedCoulombEnergy.numerical_value_in(kcal_per_mol),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(
        shiftedCoulombForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(
        wolfCoulombEnergy.numerical_value_in(kcal_per_mol),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(
        wolfCoulombForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
}

TEST(TestPairPotentialDerivatives, NonCoulombShiftedPairsAreZeroAtCutoff)
{
    constexpr auto cutoff = 4.0 * angstrom;

    const auto lennardJonesUnshifted = pot::LennardJonesPair(
        cutoff,
        LJParams{
            .c6  = -1.0 * kcal_per_mol * angstrom6,
            .c12 = 1.5 * kcal_per_mol * angstrom12
        }
    );
    const auto buckinghamUnshifted = pot::BuckinghamPair(
        cutoff,
        BuckinghamParams{
            .scaling = 2.0 * kcal_per_mol,
            .dRho    = -1.1 / angstrom,
            .c6      = -0.4 * kcal_per_mol * angstrom6
        }
    );
    const auto morseUnshifted = pot::MorsePair(
        cutoff,
        MorseParams{
            .dissociationEnergy  = 2.5 * kcal_per_mol,
            .wellWidth           = 1.4 / angstrom,
            .equilibriumDistance = 1.1 * angstrom
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
        LJParams{
            .c6  = -1.0 * kcal_per_mol * angstrom6,
            .c12 = 1.5 * kcal_per_mol * angstrom12
        }
    );
    const auto buckingham = pot::BuckinghamPair(
        cutoff,
        buckEnergyCutoff,
        buckForceCutoff,
        BuckinghamParams{
            .scaling = 2.0 * kcal_per_mol,
            .dRho    = -1.1 / angstrom,
            .c6      = -0.4 * kcal_per_mol * angstrom6
        }
    );
    const auto morse = pot::MorsePair(
        cutoff,
        morseEnergyCutoff,
        morseForceCutoff,
        MorseParams{
            .dissociationEnergy  = 2.5 * kcal_per_mol,
            .wellWidth           = 1.4 / angstrom,
            .equilibriumDistance = 1.1 * angstrom
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

    EXPECT_NEAR(ljEnergy.numerical_value_in(kcal_per_mol), 0.0, 1.0e-12);
    EXPECT_NEAR(
        ljForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(buckEnergy.numerical_value_in(kcal_per_mol), 0.0, 1.0e-12);
    EXPECT_NEAR(
        buckForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(morseEnergy.numerical_value_in(kcal_per_mol), 0.0, 1.0e-12);
    EXPECT_NEAR(
        morseForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
    EXPECT_NEAR(guffEnergy.numerical_value_in(kcal_per_mol), 0.0, 1.0e-12);
    EXPECT_NEAR(
        guffForce.numerical_value_in(kcal_per_mol / angstrom),
        0.0,
        1.0e-12
    );
}
