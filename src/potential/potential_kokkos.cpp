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

#include "potential_kokkos.hpp"

#include <cstddef>   // for size_t

#include "coulombWolf_kokkos.hpp"   // for CoulombWolf implementation with Kokkos
#include "lennardJones_kokkos.hpp"   // for LennardJones implementation with Kokkos
#include "physicalData.hpp"          // for PhysicalData
#include "simulationBox_kokkos.hpp"   // for SimulationBox implementation with Kokkos

using namespace potential;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine using Kokkos parallelization.
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine using Kokkos parallelization.
 *
 * @param simBox
 * @param physicalData
 */
void KokkosPotential::calculateForces(
    simulationBox::SimulationBox       &simBox,
    simulationBox::KokkosSimulationBox &kokkosSimBox,
    physicalData::PhysicalData         &physicalData,
    KokkosLennardJones                 &ljPotential,
    KokkosCoulombWolf                  &coulombWolf
)
{
    startTimingsSection("InterNonBonded - Transfer");

    // set total coulombic and non-coulombic energy
    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // get number of atoms
    const size_t numberOfAtoms = simBox.getNumberOfAtoms();

    kokkosSimBox.transferPositionsFromSimulationBox(simBox);
    kokkosSimBox.transferBoxDimensionsFromSimulationBox(simBox);

    const auto atomTypes       = kokkosSimBox.getAtomTypes().d_view;
    const auto moleculeIndices = kokkosSimBox.getMoleculeIndices().d_view;
    auto       internalGlobalVDWTypes =
        kokkosSimBox.getInternalGlobalVDWTypes().d_view;

    auto positions      = kokkosSimBox.getPositions().d_view;
    auto forces         = kokkosSimBox.getForces().d_view;
    auto partialCharges = kokkosSimBox.getPartialCharges().d_view;
    auto shiftForces    = kokkosSimBox.getShiftForces().d_view;
    auto boxDimensions  = kokkosSimBox.getBoxDimensions().d_view;

    const auto rcCutoff = coulombWolf.getCoulombRadiusCutOff();

    stopTimingsSection("InterNonBonded - Transfer");

    startTimingsSection("InterNonBonded");

    Kokkos::parallel_reduce(
        "Reduction",
        numberOfAtoms,
        KOKKOS_LAMBDA(
            const size_t i,
            double      &coulombEnergy,
            double      &nonCoulombEnergy
        ) {
            const auto moleculeIndex_i = moleculeIndices(i);
            const auto partialCharge_i = partialCharges(i);
            const auto vdWType_i       = internalGlobalVDWTypes(i);

            forces(i, 0) = 0.0;
            forces(i, 1) = 0.0;
            forces(i, 2) = 0.0;

            shiftForces(i, 0) = 0.0;
            shiftForces(i, 1) = 0.0;
            shiftForces(i, 2) = 0.0;

            for (size_t j = 0; j < numberOfAtoms; ++j)
            {
                const auto moleculeIndex_j = moleculeIndices(j);

                if (moleculeIndex_i == moleculeIndex_j)
                    continue;

                double dxyz[3] = {
                    positions(i, 0) - positions(j, 0),
                    positions(i, 1) - positions(j, 1),
                    positions(i, 2) - positions(j, 2)
                };

                double txyz[3];
                simulationBox::KokkosSimulationBox::calcShiftVector(
                    dxyz,
                    boxDimensions,
                    txyz
                );

                dxyz[0] += txyz[0];
                dxyz[1] += txyz[1];
                dxyz[2] += txyz[2];

                const auto distanceSquared =
                    dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];

                if (distanceSquared > rcCutoff() * rcCutoff())
                    continue;

                const auto partialCharge_j = partialCharges(j);
                const auto distance        = Kokkos::sqrt(distanceSquared);

                double force = 0.0;

                const auto coulombicEnergy = coulombWolf.calculate(
                    distance,
                    partialCharge_i,
                    partialCharge_j,
                    force
                );

                coulombEnergy += coulombicEnergy;

                const auto vdWType_j = internalGlobalVDWTypes(j);
                const auto nRCCutOff =
                    ljPotential.getRadialCutoff(vdWType_i, vdWType_j);

                if (distance < nRCCutOff)
                {
                    auto nonCoulombicEnergy =
                        ljPotential
                            .calculate(distance, force, vdWType_i, vdWType_j);
                    nonCoulombEnergy += nonCoulombicEnergy;
                }

                force /= distance;

                double force_ij[3] = {
                    force * dxyz[0],
                    force * dxyz[1],
                    force * dxyz[2]
                };

                shiftForces(i, 0) += force_ij[0] * txyz[0] / 2;
                shiftForces(i, 1) += force_ij[1] * txyz[1] / 2;
                shiftForces(i, 2) += force_ij[2] * txyz[2] / 2;

                forces(i, 0) += force_ij[0];
                forces(i, 1) += force_ij[1];
                forces(i, 2) += force_ij[2];
            }
        },
        totalCoulombEnergy,
        totalNonCoulombEnergy
    );

    stopTimingsSection("InterNonBonded");

    startTimingsSection("InterNonBonded - Transfer");

    // half energy because of double counting
    totalCoulombEnergy    *= 0.5;
    totalNonCoulombEnergy *= 0.5;

    kokkosSimBox.transferForcesToSimulationBox(simBox);
    kokkosSimBox.transferShiftForcesToSimulationBox(simBox);

    // set total coulombic and non-coulombic energy
    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    stopTimingsSection("InterNonBonded - Transfer");
}
