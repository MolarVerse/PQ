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

#include <cstddef>   // for size_t

#include "molecule.hpp"              // for Molecule
#include "physicalData.hpp"          // for PhysicalData
#include "potentialBruteForce.hpp"   // for PotentialBruteForce
#include "simulationBox.hpp"         // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Destroy the Potential Brute Force:: Potential Brute Force object
 *
 */
PotentialBruteForce::~PotentialBruteForce() = default;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialBruteForce::
    calculateForces(SimulationBox& simBox, PhysicalData& physicalData, CellList&)
{
    startTimingsSection("InterNonBonded");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    simBox.flattenForces();
    simBox.flattenShiftForces();
    simBox.flattenPositions();

    // inter molecular forces
    const size_t nMol   = simBox.getNumberOfMolecules();
    const size_t nAtoms = simBox.getNumberOfAtoms();

    const auto* const nAtomsPerMol  = simBox.getAtomsPerMoleculePtr();
    const auto* const moleculeIndex = simBox.getMoleculeIndicesPtr();
    const auto* const atomType      = simBox.getAtomTypesPtr();
    const auto* const globalVdwType = simBox.getInternalGlobalVDWTypesPtr();
    const auto* const moltype       = simBox.getMolTypesPtr();
    const auto* const pos           = simBox.getPosPtr();
    const auto* const charge        = simBox.getChargesPtr();
    auto* const       force         = simBox.getForcesPtr();
    auto* const       shiftForce    = simBox.getShiftForcesPtr();

    size_t atomIndex_i = 0;

#pragma omp target teams distribute parallel for collapse(2)
    for (size_t atomIndex_i = 0; atomIndex_i < nAtoms; ++atomIndex_i)
    {
        for (size_t atomIndex_j = 0; atomIndex_j < nAtoms; ++atomIndex_j)
        {
            const size_t mol_i = moleculeIndex[atomIndex_i];
            const size_t mol_j = moleculeIndex[atomIndex_j];

            if (mol_i < mol_j)
            {
                Real fx = 0, fy = 0, fz = 0;
                Real shiftfx = 0, shiftfy = 0, shiftfz = 0;

                const auto [coulombEnergy, nonCoulombEnergy] =
                    calculateSingleInteraction(
                        *box,
                        pos[atomIndex_i * 3],
                        pos[atomIndex_i * 3 + 1],
                        pos[atomIndex_i * 3 + 2],
                        pos[atomIndex_j * 3],
                        pos[atomIndex_j * 3 + 1],
                        pos[atomIndex_j * 3 + 2],
                        atomType[atomIndex_i],
                        atomType[atomIndex_j],
                        globalVdwType[atomIndex_i],
                        globalVdwType[atomIndex_j],
                        moltype[mol_i],
                        moltype[mol_j],
                        charge[atomIndex_i],
                        charge[atomIndex_j],
                        fx,
                        fy,
                        fz,
                        shiftfx,
                        shiftfy,
                        shiftfz
                    );

                // clang-format off
                    #pragma omp atomic
                    force[atomIndex_i * 3]     += fx;
                    #pragma omp atomic
                    force[atomIndex_i * 3 + 1] += fy;
                    #pragma omp atomic
                    force[atomIndex_i * 3 + 2] += fz;

                    #pragma omp atomic
                    force[atomIndex_j * 3]     -= fx;
                    #pragma omp atomic
                    force[atomIndex_j * 3 + 1] -= fy;
                    #pragma omp atomic
                    force[atomIndex_j * 3 + 2] -= fz;

                    #pragma omp atomic
                    shiftForce[atomIndex_i * 3]     += shiftfx;
                    #pragma omp atomic
                    shiftForce[atomIndex_i * 3 + 1] += shiftfy;
                    #pragma omp atomic
                    shiftForce[atomIndex_i * 3 + 2] += shiftfz;

                    #pragma omp atomic
                    totalCoulombEnergy    += coulombEnergy;
                    #pragma omp atomic
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
        }
    }
    // clang-format on

    simBox.deFlattenForces();
    simBox.deFlattenShiftForces();

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief clone the potential
 *
 * @return std::shared_ptr<PotentialBruteForce>
 */
std::shared_ptr<Potential> PotentialBruteForce::clone() const
{
    return std::make_shared<PotentialBruteForce>(*this);
}