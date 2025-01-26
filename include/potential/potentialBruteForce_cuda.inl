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

#ifndef __POTENTIAL_BRUTE_FORCE_INL__
#define __POTENTIAL_BRUTE_FORCE_INL__

/**
 * @file potentialBruteForce.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the main implementation of the brute force
 * potential calculation. The brute force calculation routine is only used to
 * determine all inter non-bonded forces and energies. The calculation is done
 * in a brute force manner, were only the upper diagonal matrix of all atom
 * interactions is considered excluding all interactions of atoms belonging to
 * same molecule.
 *
 * @date 2024-12-09
 *
 */

#include <cassert>

#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "debug.hpp"
#include "lennardJones.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "triclinicBox.hpp"

namespace potential
{
    /**
     * @brief a template function to calculate the brute force approach for the
     * non-bonded interactions.
     *
     * @details The function calculates the non-bonded forces and energies for
     * all atom pairs. The function is implemented in a brute force manner and
     * is only used to determine all inter non-bonded forces and energies. The
     * calculation is done in a brute force manner, were only the upper diagonal
     * matrix of all atom interactions is considered excluding all interactions
     * of atoms belonging to same molecule.
     *
     * @tparam CoulombType the type of the coulomb potential
     * @tparam NonCoulombType the type of the non-coulomb potential
     * @tparam BoxType the type of the simulation box
     * @param pos the positions of all atoms
     * @param force the forces acting on all atoms
     * @param shiftForce the forces acting on all atoms due to the shift
     * @param charge the charges of all atoms
     * @param coulParams the parameters of the coulomb potential
     * @param nonCoulParams the parameters of the non-coulomb potential
     * @param ncCutOffs the cutoffs of the non-coulomb potential
     * @param boxParams the parameters of the simulation box
     * @param moleculeIndex the index of the molecule each atom belongs to
     * @param molTypes the types of the molecules
     * @param atomTypes the types of the atoms
     * @param totalCoulombEnergy the total coulomb energy
     * @param totalNonCoulombEnergy the total non-coulomb energy
     * @param coulCutOff the cutoff of the coulomb potential
     * @param nAtoms the number of atoms
     * @param nAtomTypes the number of atom types
     * @param nonCoulParamsOffset the offset of the non-coulomb parameters
     * @param maxNumAtomTypes the maximum number of atom types
     * @param nMolTypes the number of molecule types
     */
    template <typename CoulombType, typename NonCoulombType, typename BoxType>
    __global__ void bruteForce(
        const Real* __restrict__ const pos,
        Real* __restrict__ const force,
        Real* __restrict__ const shiftForce,
        const Real* __restrict__ const charge,
        const Real* __restrict__ const coulParams,
        const Real* __restrict__ const nonCoulParams,
        const Real* __restrict__ const ncCutOffs,
        const Real* __restrict__ const boxParams,
        const size_t* __restrict__ const moleculeIndex,
        const size_t* __restrict__ const molTypes,
        const size_t* __restrict__ const atomTypes,
        Real* __restrict__ totalCoulombEnergy,
        Real* __restrict__ totalNonCoulombEnergy,
        const Real   coulCutOff,
        const size_t nAtoms,
        const size_t nAtomTypes,
        const size_t nonCoulParamsOffset,
        const size_t maxNumAtomTypes,
        const size_t nMolTypes
    )
    {
        const auto ithread = threadIdx.x + blockIdx.x * blockDim.x;
        const auto iblock  = blockIdx.x;
        const auto nblocks = gridDim.x;

        const auto coulCutOffSquared = coulCutOff * coulCutOff;
        // const auto indexHelper1      = nMolTypes * nMolTypes *
        // maxNumAtomTypes; const auto indexHelper2      = nMolTypes *
        // maxNumAtomTypes;

        for (size_t i = ithread; i < nAtoms; i += blockDim.x * nblocks)
        {
            Real fx = 0;
            Real fy = 0;
            Real fz = 0;

            Real shiftFx = 0;
            Real shiftFy = 0;
            Real shiftFz = 0;

            Real coulombEnergy_i    = 0.0;
            Real nonCoulombEnergy_i = 0.0;

            const auto xi = pos[i * 3];
            const auto yi = pos[i * 3 + 1];
            const auto zi = pos[i * 3 + 2];

            const auto charge_i = charge[i];

            const auto mol_i = moleculeIndex[i];

            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto mol_j = moleculeIndex[j];

                if (mol_i == mol_j)
                    continue;

                Real dx = xi - pos[j * 3];
                Real dy = yi - pos[j * 3 + 1];
                Real dz = zi - pos[j * 3 + 2];

                Real tx = 0;
                Real ty = 0;
                Real tz = 0;

                image<BoxType>(boxParams, dx, dy, dz, tx, ty, tz);

                const double distanceSquared = dx * dx + dy * dy + dz * dz;

                if (distanceSquared < coulCutOffSquared)
                {
                    Real localForce    = 0;
                    auto coulombEnergy = 0.0;

                    const auto distance = ::sqrt(distanceSquared);

                    const auto coulombPreFactor = charge_i * charge[j];

                    calculateCoulombPotential<CoulombType>(
                        coulombEnergy,
                        localForce,
                        distance,
                        coulombPreFactor,
                        coulCutOff,
                        coulParams
                    );

                    coulombEnergy_i += coulombEnergy;

                    int combinedCutoffIndex;
                    int combinedIndex;

                    fetchCombinedIndices<NonCoulombType>(
                        atomTypes,
                        molTypes,
                        mol_i,
                        mol_j,
                        i,
                        j,
                        nAtomTypes,
                        nMolTypes,
                        nonCoulParamsOffset,
                        combinedCutoffIndex,
                        combinedIndex
                    );

                    const auto rncCutOff = ncCutOffs[combinedCutoffIndex];

                    if (distance < rncCutOff)
                    {
                        auto nonCoulombEnergy = 0.0;

                        calculateNonCoulombEnergy<NonCoulombType>(
                            nonCoulombEnergy,
                            localForce,
                            distance,
                            distanceSquared,
                            rncCutOff,
                            &nonCoulParams[combinedIndex]
                        );

                        nonCoulombEnergy_i += nonCoulombEnergy;
                    }

                    localForce /= distance;

                    fx += localForce * dx;
                    fy += localForce * dy;
                    fz += localForce * dz;

                    shiftFx += localForce * dx * tx;
                    shiftFy += localForce * dy * ty;
                    shiftFz += localForce * dz * tz;
                }
            }

            atomicAdd(force + i * 3, +fx);
            atomicAdd(force + i * 3 + 1, +fy);
            atomicAdd(force + i * 3 + 2, +fz);

            atomicAdd(shiftForce + i * 3, +shiftFx * 0.5);
            atomicAdd(shiftForce + i * 3 + 1, +shiftFy * 0.5);
            atomicAdd(shiftForce + i * 3 + 2, +shiftFz * 0.5);

            atomicAdd(totalCoulombEnergy, coulombEnergy_i * 0.5);
            atomicAdd(totalNonCoulombEnergy, nonCoulombEnergy_i * 0.5);
        }
    }

}   // namespace potential

#endif   // __POTENTIAL_BRUTE_FORCE_INL__