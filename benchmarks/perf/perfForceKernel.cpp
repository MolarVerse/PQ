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

// Fixed-work micro-benchmark of the non-bonded per-pair force kernel
// (Coulomb + non-Coulomb pair evaluation, the per-pair getNonCoulPair lookup
// and force accumulation), so callgrind yields a stable instruction count.

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "coulombShiftedPotential.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    auto molecule            = benchSetup::makeMolecule({.nAtoms = 2});
    auto nonCoulombPotential = benchSetup::makeNonCoulomb();
    auto coulombPotential    = pot::CoulombShiftedPotential(10.0);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.75);

    auto intraNonBondedType =
        intraNonBonded::IntraNonBondedContainer(0, {{-1}});
    auto intraNonBondedMap =
        intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto simulationBox = molsys::SimulationBox();
    simulationBox.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData = physicalData::PhysicalData();

    const auto box     = simulationBox.getBoxDimensions();
    const auto atomIdx = intraNonBondedType.getAtomIndices()[0][0];

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        const auto [coulombEnergy, nonCoulombEnergy] =
            intraNonBondedMap.calculateSingleInteraction(
                0,
                atomIdx,
                box,
                physicalData,
                &coulombPotential,
                &nonCoulombPotential
            );

        sink += coulombEnergy + nonCoulombEnergy;
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
