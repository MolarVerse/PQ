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

#include "settings.hpp"
#include "units.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "angleForceField.hpp"
#include "bondForceField.hpp"
#include "coulombShiftedPotential.hpp"
#include "dihedralForceField.hpp"
#include "perfAlloc.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    Settings settings;
    settings.potentialSettings.setCoulombRadiusCutOffByUser(20.0 * angstrom);
    auto box = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    auto physicalData = physicalData::PhysicalData();
    auto coulombPotential =
        pot::CoulombShiftedPotential(settings.potentialSettings);
    auto nonCoulombPotential = benchSetup::makeNonCoulomb();

    auto molecule = benchSetup::makeMolecule({.nAtoms = 4});

    PotentialSettings::setScale14Coulomb(0.75);
    PotentialSettings::setScale14VanDerWaals(0.5);

    auto bond = ff::BondForceField(
        &molecule,
        &molecule,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    bond.setParams(
        BondParams{
            .equilibrium   = 1.2 * angstrom,
            .forceConstant = 3.0 * kcal_per_mol / angstrom / angstrom
        }
    );

    auto angle = ff::AngleForceField(
        {&molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    angle.setParams(
        AngleParams{
            .equilibrium   = M_PI / 2.0 * rad,
            .forceConstant = 3.0 * kcal_per_mol / rad / rad
        }
    );

    auto dihedral = ff::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    dihedral.setParams(
        DihedralParams{
            .forceConstant = 3.0 * kcal_per_mol,
            .frequency     = 3,
            .phaseShift    = M_PI * rad
        }
    );
    dihedral.setIsLinker(false);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    const auto smoothing = settings.hybridSettings.getSmoothingMethod();

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        bond.calculateEnergyAndForces(
            box,
            physicalData,
            coulombPotential,
            nonCoulombPotential,
            smoothing,
            settings.potentialSettings
        );
        angle.calculateEnergyAndForces(
            box,
            physicalData,
            coulombPotential,
            nonCoulombPotential,
            smoothing,
            settings.potentialSettings
        );
        dihedral.calculateEnergyAndForces(
            box,
            physicalData,
            false,
            coulombPotential,
            nonCoulombPotential,
            smoothing,
            settings.potentialSettings
        );
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format(
        "{:.6f}\n",
        (physicalData.getBondEnergy() + physicalData.getAngleEnergy() +
         physicalData.getDihedralEnergy())
                .numerical_value_in(kcal_per_mol) +
            molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
                kcal_per_mol / angstrom
            )
    );
    return 0;
}
