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
// dihedral).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "angleForceField.hpp"
#include "bondForceField.hpp"
#include "coulombShiftedPotential.hpp"
#include "dihedralForceField.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(20.0);
    auto nonCoulombPotential = benchSetup::makeNonCoulomb();

    auto molecule = benchSetup::makeMolecule({.nAtoms = 4});

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    auto bond =
        forceField::BondForceField(&molecule, &molecule, 0, 1, BondId{0});
    bond.setEquilibriumBondLength(1.2);
    bond.setForceConstant(3.0);

    auto angle = forceField::AngleForceField(
        {&molecule, &molecule, &molecule},
        {0, 1, 2},
        AngleId{0}
    );
    angle.setEquilibriumAngle(M_PI / 2.0);
    angle.setForceConstant(3.0);

    auto dihedral = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {0, 1, 2, 3},
        DihedralId{0}
    );
    dihedral.setPhaseShift(M_PI);
    dihedral.setPeriodicity(3);
    dihedral.setForceConstant(3.0);
    dihedral.setIsLinker(false);

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
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
    std::cout << std::format(
        "{:.6f}\n",
        physicalData.getBondEnergy() + physicalData.getAngleEnergy() +
            physicalData.getDihedralEnergy() + molecule.getAtomForce(0)[0]
    );
    return 0;
}
