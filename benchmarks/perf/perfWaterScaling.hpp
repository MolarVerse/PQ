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

#ifndef PQ_BENCH_WATER_SCALING_HPP
#define PQ_BENCH_WATER_SCALING_HPP

// Shared body of the cell list / inter-water scaling benchmarks: a cubic grid
// of perSide^3 water molecules (4.5 A spacing, so the density is the same for
// every size), the number of cells grows with the box (cell edge >= cutoff).
// Each iteration updates the cell list and computes the inter-water forces, so
// the cost per iteration should grow linearly with the number of molecules.

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "celllist.hpp"
#include "coulombShiftedPotential.hpp"
#include "generalSettings.hpp"
#include "interWater.hpp"
#include "lennardJonesPair.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"
#include "waterModelSettings.hpp"

namespace benchWaterScaling
{
    inline constexpr MolType WATER_TYPE{1};
    inline constexpr double  CUTOFF                 = 9.0;
    inline constexpr double  SPACING                = 4.5;
    inline constexpr auto    HYDROGEN_ATOMIC_NUMBER = AtomNumber{1};
    inline constexpr auto    OXYGEN_ATOMIC_NUMBER   = AtomNumber{8};

    inline int run(const std::size_t perSide, const std::uint64_t iterations)
    {
        settings::PotentialSettings::setCoulombRadiusCutOff(CUTOFF);
        pot::CoulombPotential::setCoulombRadiusCutOff(CUTOFF);
        pot::CoulombPotential::setCoulombEnergyCutOff(0.0);
        pot::CoulombPotential::setCoulombForceCutOff(0.0);

        WaterModelSettings waterModelSettings;
        waterModelSettings.setWaterInterModel(WaterInterModel::SPC_E);

        const double boxLength = (SPACING * static_cast<double>(perSide)) + 3.0;
        const auto   numberOfCells = static_cast<size_t>(boxLength / CUTOFF);

        molsys::SimulationBox   simBox;
        molsys::OrthorhombicBox orthoBox{{boxLength, boxLength, boxLength}};
        simBox.setBox(orthoBox, false);
        simBox.setWaterType(WATER_TYPE);

        molsys::MoleculeType waterType;
        waterType.setMoltype(WATER_TYPE);
        waterType.setNumberOfAtoms(3);
        simBox.addMoleculeType(waterType);

        const auto makeAtom = [](const std::string_view name,
                                 const linalg::Vec3D   &pos,
                                 const double           charge,
                                 const AtomNumber       atomicNumber)
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setName(name);
            atom->setAtomicNumber(atomicNumber);
            atom->setPosition(pos);
            atom->setAtomType(AtomType{0});
            atom->setInternalGlobalVDWType(VdwType{0});
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
                    const linalg::Vec3D oxygen{
                        1.0 + (SPACING * static_cast<double>(ix)),
                        1.0 + (SPACING * static_cast<double>(iy)),
                        1.0 + (SPACING * static_cast<double>(iz))
                    };

                    molsys::Molecule molecule;
                    molecule.setMoltype(WATER_TYPE);
                    molecule.addAtom(
                        makeAtom("O", oxygen, -0.82, OXYGEN_ATOMIC_NUMBER)
                    );
                    molecule.addAtom(makeAtom(
                        "H",
                        oxygen + linalg::Vec3D(0.9572, 0.0, 0.0),
                        0.41,
                        HYDROGEN_ATOMIC_NUMBER
                    ));
                    molecule.addAtom(makeAtom(
                        "H",
                        oxygen + linalg::Vec3D(-0.24, 0.927, 0.0),
                        0.41,
                        HYDROGEN_ATOMIC_NUMBER
                    ));
                    simBox.addMolecule(molecule);
                }
            }
        }

        waterModel::InterWaterState state;
        state.oxygenCharge     = -0.82;
        state.hydrogenCharge   = 0.41;
        state.nonCoulombPairOO = std::make_unique<pot::LennardJonesPair>(
            CUTOFF,
            LJParams{.c6 = 2.0, .c12 = 4.0}
        );
        state.nonCoulombPairOH = std::make_unique<pot::LennardJonesPair>(
            CUTOFF,
            LJParams{.c6 = 0.5, .c12 = 1.5}
        );
        state.nonCoulombPairHH = std::make_unique<pot::LennardJonesPair>(
            CUTOFF,
            LJParams{.c6 = 0.2, .c12 = 0.8}
        );

        waterModel::InterWater interWater(
            std::move(state),
            std::make_unique<waterModel::InterWaterStrategyCellList>()
        );
        auto coulombPot =
            std::make_shared<pot::CoulombShiftedPotential>(CUTOFF);

        settings::GeneralSettings::activateCellList();
        molsys::CellList cellList;
        cellList.setNumberOfCells(numberOfCells);
        cellList.resizeCells();
        cellList.setup(simBox);

        auto physicalData = physicalData::PhysicalData();

        CALLGRIND_ZERO_STATS;

        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            cellList.updateCellList(simBox);
            interWater.calculate(simBox, physicalData, coulombPot, cellList);
        }

        // read state so the loop cannot be optimized away
        std::cout << std::fixed << std::setprecision(6)
                  << physicalData.getCoulombEnergy() << " "
                  << physicalData.getNonCoulombEnergy() << "\n";
        return 0;
    }
}   // namespace benchWaterScaling

#endif   // PQ_BENCH_WATER_SCALING_HPP
