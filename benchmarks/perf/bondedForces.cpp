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

// Fixed-work micro-benchmark of the bonded force kernels (bond, angle,
// dihedral). Setup mirrors the forceField unit tests.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "angleForceField.hpp"
#include "atom.hpp"
#include "bondForceField.hpp"
#include "coulombShiftedPotential.hpp"
#include "dihedralForceField.hpp"
#include "forceFieldNonCoulomb.hpp"
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

static constexpr long ITERATIONS = 20000;

int main()
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(20.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair =
        potential::LennardJonesPair(size_t(0), size_t(1), 15.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    nonCoulombPotential.setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto molecule = simulationBox::Molecule();
    molecule.setMoltype(0);
    molecule.setNumberOfAtoms(4);

    for (size_t i = 0; i < 4; ++i)
    {
        auto atom = std::make_shared<simulationBox::Atom>();
        atom->setPosition({double(i), 0.5 * double(i), 0.3 * double(i)});
        atom->setForce({0.0, 0.0, 0.0});
        atom->setInternalGlobalVDWType(i % 2);
        atom->setAtomType(i % 2);
        atom->setPartialCharge((i % 2 == 0) ? 1.0 : -0.5);
        molecule.addAtom(atom);
    }

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    auto bond = forceField::BondForceField(&molecule, &molecule, 0, 1, 0);
    bond.setEquilibriumBondLength(1.2);
    bond.setForceConstant(3.0);

    auto angle =
        forceField::AngleForceField({&molecule, &molecule, &molecule}, {0, 1, 2}, 0);
    angle.setEquilibriumAngle(M_PI / 2.0);
    angle.setForceConstant(3.0);

    auto dihedral = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {0, 1, 2, 3},
        0
    );
    dihedral.setPhaseShift(M_PI);
    dihedral.setPeriodicity(3);
    dihedral.setForceConstant(3.0);
    dihedral.setIsLinker(false);

    CALLGRIND_ZERO_STATS;

    for (long i = 0; i < ITERATIONS; ++i)
    {
        bond.calculateEnergyAndForces(
            box,
            physicalData,
            coulombPotential,
            nonCoulombPotential
        );
        angle.calculateEnergyAndForces(
            box,
            physicalData,
            coulombPotential,
            nonCoulombPotential
        );
        dihedral.calculateEnergyAndForces(
            box,
            physicalData,
            false,
            coulombPotential,
            nonCoulombPotential
        );
    }

    // read state so the loop cannot be optimized away
    std::printf(
        "%.6f\n",
        physicalData.getBondEnergy() + physicalData.getAngleEnergy() +
            physicalData.getDihedralEnergy() + molecule.getAtomForce(0)[0]
    );
    return 0;
}
