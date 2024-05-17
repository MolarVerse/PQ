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

#include "molecule.hpp"       // for Molecule
#include "physicalData.hpp"   // for PhysicalData
#include "potential.hpp"
#include "simulationBox_kokkos.hpp"   // for SimulationBox implementation with Kokkos
#include "coulombPotential.hpp"      // for CoulombPotential

namespace simulationBox
{
    class CellList;
}   // namespace simulationBox

using namespace potential;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine using Kokkos parallelization.
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialKokkos::
    calculateForces(simulationBox::SimulationBox &simBox, simulationBox::KokkosSimulationBox &kokkosSimBox, physicalData::PhysicalData &physicalData, simulationBox::CellList &)
{
    Kokkos::initialize();
    {
        // set total coulombic and non-coulombic energy
        double totalCoulombEnergy    = 0.0;
        double totalNonCoulombEnergy = 0.0;

        // get number of atoms
        const size_t numberOfAtoms = simBox.getNumberOfAtoms();

        kokkosSimBox.initializeForces();
        kokkosSimBox.transferPositionsFromSimulationBox(simBox);

        auto atomTypes = kokkosSimBox.getAtomTypes().d_view;
        auto molTypes  = kokkosSimBox.getMolTypes().d_view;
        auto internalGlobalVDWTypes =
            kokkosSimBox.getInternalGlobalVDWTypes().d_view;

        auto positions      = kokkosSimBox.getPositions().d_view;
        auto forces         = kokkosSimBox.getForces().d_view;
        auto partialCharges = kokkosSimBox.getPartialCharges().d_view;

        Kokkos::parallel_reduce(
            "Reduction",
            numberOfAtoms,
            KOKKOS_LAMBDA(
                const size_t i,
                double      &coulombEnergy,
                double      &nonCoulombEnergy
            ) {

                const auto partialCharge_i = partialCharges(i);
                const auto vdWType_i       = internalGlobalVDWTypes(i);
                auto force_i               = &forces(i, 0);

                for (size_t j = 0; j < numberOfAtoms; ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    double dxyz[3] = {
                        positions(i, 0) - positions(j, 0),
                        positions(i, 1) - positions(j, 1),
                        positions(i, 2) - positions(j, 2)
                    };

                    auto normSquared = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];
                    auto distance   = Kokkos::sqrt(normSquared);


                    if (distance < CoulombPotential::getCoulombRadiusCutOff())
                    {
                        continue;
                    }

                    const auto partialCharge_j = partialCharges(j);
                    const auto vdWType_j       = internalGlobalVDWTypes(j);

                    auto [coulombicEnergy, nonCoulombicEnergy] =
                        PotentialKokkos::calculatePairEnergy
                        (
                            distance,
                            dxyz,
                            force_i,
                            partialCharge_i,
                            vdWType_i,
                            partialCharge_j,
                            vdWType_j
                        );

                    coulombEnergy    += coulombicEnergy;
                    nonCoulombEnergy += nonCoulombicEnergy;
                }
            },
            totalCoulombEnergy,
            totalNonCoulombEnergy
        );

        // set total coulombic and non-coulombic energy
        physicalData.setCoulombEnergy(totalCoulombEnergy);
        physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    }   // end of Kokkos scope
    Kokkos::finalize();
}

/**
 * @brief calculates pair energy for two atoms
 *
 * @return Kokkos::pair<double, double>
 */
KOKKOS_INLINE_FUNCTION
Kokkos::pair<double, double> PotentialKokkos::calculatePairEnergy(
    const double distance,
    const double dxyz[3],
    double *force_i,
    const double partialCharge_i,
    const size_t vdWType_i,
    const double partialCharge_j,
    const size_t vdWType_j
)
{
    auto coulombicEnergy    = 0.0;
    auto nonCoulombicEnergy = 0.0;



    return Kokkos::make_pair(coulombicEnergy, nonCoulombicEnergy);
}
