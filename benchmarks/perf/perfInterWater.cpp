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

// Fixed-work micro-benchmark of the intermolecular water kernel (cell list).

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "celllist.hpp"
#include "coulombShiftedPotential.hpp"
#include "interWater.hpp"
#include "lennardJonesPair.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"
#include "waterModelSettings.hpp"

using namespace simulationBox;
using namespace potential;
using namespace waterModel;
using linearAlgebra::Vec3D;

static constexpr std::uint64_t ITERATIONS             = 50;
static constexpr size_t        WATER_TYPE             = 1;
static constexpr double        CUTOFF                 = 9.0;
static constexpr int           HYDROGEN_ATOMIC_NUMBER = 1;
static constexpr int           OXYGEN_ATOMIC_NUMBER   = 8;

int main()
{
    settings::PotentialSettings::setCoulombRadiusCutOff(CUTOFF);
    CoulombPotential::setCoulombRadiusCutOff(CUTOFF);
    CoulombPotential::setCoulombEnergyCutOff(0.0);
    CoulombPotential::setCoulombForceCutOff(0.0);
    settings::WaterModelSettings::setIsInterWaterModelSet(true);

    // ~6x6x6 = 216 water molecules (648 atoms) on a 3 Å grid in a 30 Å box.
    constexpr size_t perSide = 6;
    constexpr double spacing = 4.5;

    SimulationBox simBox;
    simBox.setBoxDimensions({30.0, 30.0, 30.0});
    simBox.setWaterType(WATER_TYPE);

    MoleculeType waterType;
    waterType.setMoltype(WATER_TYPE);
    waterType.setNumberOfAtoms(3);
    simBox.addMoleculeType(waterType);

    const auto makeAtom = [](const std::string_view name,
                             const Vec3D           &pos,
                             const double           charge,
                             const int              atomicNumber)
    {
        auto atom = std::make_shared<Atom>();
        atom->setName(name);
        atom->setAtomicNumber(atomicNumber);
        atom->setPosition(pos);
        atom->setAtomType(0);
        atom->setInternalGlobalVDWType(0);
        atom->setPartialCharge(charge);
        atom->setForceToZero();
        return atom;
    };

    for (size_t ix = 0; ix < perSide; ++ix)
    {
        for (size_t iy = 0; iy < perSide; ++iy)
        {
            for (size_t iz = 0; iz < perSide; ++iz)
            {
                const Vec3D o{
                    1.0 + spacing * static_cast<double>(ix),
                    1.0 + spacing * static_cast<double>(iy),
                    1.0 + spacing * static_cast<double>(iz)
                };

                Molecule molecule;
                molecule.setMoltype(WATER_TYPE);
                molecule.setNumberOfAtoms(3);
                molecule.addAtom(makeAtom("O", o, -0.82, OXYGEN_ATOMIC_NUMBER));
                molecule.addAtom(makeAtom(
                    "H",
                    o + Vec3D(0.9572, 0.0, 0.0),
                    0.41,
                    HYDROGEN_ATOMIC_NUMBER
                ));
                molecule.addAtom(makeAtom(
                    "H",
                    o + Vec3D(-0.24, 0.927, 0.0),
                    0.41,
                    HYDROGEN_ATOMIC_NUMBER
                ));
                simBox.addMolecule(molecule);
            }
        }
    }

    InterWaterState state;
    state._oxygenCharge   = -0.82;
    state._hydrogenCharge = 0.41;
    state._nonCoulombPairOO =
        std::make_unique<LennardJonesPair>(CUTOFF, -2.0, 4.0);
    state._nonCoulombPairOH =
        std::make_unique<LennardJonesPair>(CUTOFF, -0.5, 1.5);
    state._nonCoulombPairHH =
        std::make_unique<LennardJonesPair>(CUTOFF, -0.2, 0.8);

    InterWater interWater(
        std::move(state),
        std::make_unique<InterWaterStrategyCellList>()
    );

    auto coulombPot = std::make_shared<CoulombShiftedPotential>(CUTOFF);

    CellList cellList;
    cellList.setNumberOfCells(3);
    cellList.resizeCells();
    cellList.setup(simBox);
    cellList.activate();
    cellList.updateCellList(simBox);

    auto physicalData = physicalData::PhysicalData();

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
        interWater.calculate(simBox, physicalData, coulombPot, cellList);

    // read state so the loop cannot be optimized away
    std::cout << std::fixed << std::setprecision(6)
              << physicalData.getCoulombEnergy() << " "
              << physicalData.getNonCoulombEnergy() << "\n";
    return 0;
}
