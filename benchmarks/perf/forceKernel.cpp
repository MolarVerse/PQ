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

// Fixed-work micro-benchmark of the non-bonded per-pair force kernel.
//
// It does a deterministic, fixed number of iterations of the inner-loop work
// (Coulomb + non-Coulomb pair evaluation, the per-pair getNonCoulPair lookup
// and force accumulation), so that running it under callgrind yields a stable
// instruction count usable as a CI performance-regression gate.
//
// The setup mirrors tests/src/intraNonBonded/testIntraNonBondedMap.cpp.

#include <cstddef>
#include <cstdio>
#include <memory>

// callgrind hook: zero the instruction counter after one-time setup so the
// reported count reflects only the measured loop. The macro is a no-op outside
// valgrind, and compiles to nothing when the header is unavailable.
#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

// number of kernel evaluations; fixed so the instruction count is comparable
// across builds. Small is fine: callgrind counts are deterministic and we zero
// the counter after setup, so the signal is pure loop work.
static constexpr long ITERATIONS = 20000;

int main()
{
    auto molecule = simulationBox::Molecule(0);
    molecule.setNumberOfAtoms(2);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({0.0, 0.0, 11.0});
    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom1->setInternalGlobalVDWType(0);
    atom2->setInternalGlobalVDWType(1);
    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom1->setPartialCharge(0.5);
    atom2->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.75);

    auto intraNonBondedType =
        intraNonBonded::IntraNonBondedContainer(0, {{-1}});
    auto intraNonBondedMap =
        intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();
    nonCoulombPotential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );

    auto nonCoulombPair =
        potential::LennardJonesPair(size_t(0), size_t(1), 10.0, 2.0, 3.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    nonCoulombPotential.setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto simulationBox = simulationBox::SimulationBox();
    simulationBox.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData = physicalData::PhysicalData();

    const auto box     = simulationBox.getBoxDimensions();
    const auto atomIdx = intraNonBondedType.getAtomIndices()[0][0];

    // exclude the one-time setup above from the measured instruction count
    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (long i = 0; i < ITERATIONS; ++i)
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

    // print so the loop cannot be optimized away
    std::printf("%.6f\n", sink);
    return 0;
}
