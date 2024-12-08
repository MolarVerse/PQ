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

#ifndef __POTENTIAL_BRUTE_FORCE_TPP__
#define __POTENTIAL_BRUTE_FORCE_TPP__

#include "coulombShiftedPotential.hpp"
#include "debug.hpp"
#include "lennardJones.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "triclinicBox.hpp"

namespace potential
{

    template <typename CoulombType, typename NonCoulombType, typename BoxType>
    void bruteForce(
        const Real* const   pos,
        Real* const         force,
        Real* const         shiftForce,
        const Real* const   charge,
        const Real* const   coulParams,
        const Real* const   nonCoulParams,
        const Real* const   ncCutOffs,
        const Real* const   boxParams,
        const size_t* const moleculeIndex,
        const size_t* const molTypes,
        const size_t* const atomTypes,
        Real&               totalCoulombEnergy,
        Real&               totalNonCoulombEnergy,
        const Real          coulCutOff,
        const size_t        nAtoms,
        const size_t        nAtomTypes,
        const size_t        nonCoulParamsOffset
    )
    {
        __DEBUG_LOCATION__();

        const auto coulCutOffSquared = coulCutOff * coulCutOff;

        // clang-format off
        #pragma omp target teams distribute parallel for collapse(2)
        for (size_t atomIndex_i = 0; atomIndex_i < nAtoms; ++atomIndex_i)
        {
            for (size_t atomIndex_j = 0; atomIndex_j < nAtoms; ++atomIndex_j)
            {
                const size_t mol_i = moleculeIndex[atomIndex_i];
                const size_t mol_j = moleculeIndex[atomIndex_j];

                if (mol_i < mol_j)
                {
                    const auto xi = pos[atomIndex_i * 3];
                    const auto yi = pos[atomIndex_i * 3 + 1];
                    const auto zi = pos[atomIndex_i * 3 + 2];

                    const auto xj = pos[atomIndex_j * 3];
                    const auto yj = pos[atomIndex_j * 3 + 1];
                    const auto zj = pos[atomIndex_j * 3 + 2];

                    auto dx = xi - xj;
                    auto dy = yi - yj;
                    auto dz = zi - zj;

                    Real tx = 0;
                    Real ty = 0;
                    Real tz = 0;

                    if constexpr (std::is_same_v<BoxType, simulationBox::OrthorhombicBox>)
                        simulationBox::imageOrthoRhombic(
                            boxParams,
                            dx,
                            dy,
                            dz,
                            tx,
                            ty,
                            tz
                        );
                    else
                        throw customException::NotImplementedException(
                            "The triclinic box is not implemented yet"
                        );

                    const double distanceSquared = dx * dx + dy * dy + dz * dz;

                    if(distanceSquared < coulCutOffSquared)
                    {
                        Real localForce = 0;
                        Real fx = 0;
                        Real fy = 0;
                        Real fz = 0;
                        Real shiftFx = 0;
                        Real shiftFy = 0;
                        Real shiftFz = 0;

                        auto coulombEnergy    = 0.0;

                        const double distance = ::sqrt(distanceSquared);

                        const auto coulombPreFactor = charge[atomIndex_i] * charge[atomIndex_j];

                        if constexpr (std::is_same_v<CoulombType, CoulombShiftedPotential>)
                        {
                            coulombEnergy = calculateCoulombShiftedPotential(
                                localForce,
                                distance,
                                coulombPreFactor,
                                coulCutOff,
                                coulParams
                            );
                        }

                        #pragma omp atomic
                        totalCoulombEnergy    += coulombEnergy;

                        auto combinedCutoffIndex = -1;
                        auto combinedIndex      = -1;

                        // TODO: check here all FF types combined
                        if constexpr (std::is_same_v<NonCoulombType, LennardJonesFF>){
                            combinedCutoffIndex = atomTypes[atomIndex_i] * nAtomTypes + atomTypes[atomIndex_j];
                            combinedIndex = atomTypes[atomIndex_i] * nAtomTypes * nonCoulParamsOffset + atomTypes[atomIndex_j] * nonCoulParamsOffset;
                        }
                        else
                            throw customException::NotImplementedException(
                                "The nonCoulomb potential is not implemented yet"
                            );

                        const auto rncCutOff = ncCutOffs[combinedCutoffIndex];

                        if (distance < rncCutOff)
                        {
                            auto nonCoulombEnergy = 0.0;
                            if constexpr (std::is_same_v<NonCoulombType, LennardJonesFF>)
                            {
                                nonCoulombEnergy = calculateLennardJones(
                                    localForce,
                                    distance,
                                    distanceSquared,
                                    rncCutOff,
                                    &nonCoulParams[combinedIndex]
                                );
                            }

                            #pragma omp atomic
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }

                        fx = localForce / distance * dx;
                        fy = localForce / distance * dy;
                        fz = localForce / distance * dz;

                        shiftFx = fx * tx;
                        shiftFy = fy * ty;
                        shiftFz = fz * tz;

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
                        shiftForce[atomIndex_i * 3]     += shiftFx;
                        #pragma omp atomic
                        shiftForce[atomIndex_i * 3 + 1] += shiftFy;
                        #pragma omp atomic
                        shiftForce[atomIndex_i * 3 + 2] += shiftFz;
                    }
                }
            }
        }
        // clang-format on
    }
}   // namespace potential

#endif   // __POTENTIAL_BRUTE_FORCE_TPP__