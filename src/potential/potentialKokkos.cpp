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
#include "simulationBox.hpp"   // for SimulationBox

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
    calculateForces(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData, simulationBox::CellList &)
{
    // Kokkos::initialize();
    // {
    //     // set total coulombic and non-coulombic energy
    //     double totalCoulombEnergy    = 0.0;
    //     double totalNonCoulombEnergy = 0.0;

    //     // get number of atoms
    //     const size_t numberOfAtoms = simBox.getNumberOfAtoms();

    //     // create Kokkos Views for positions and forces
    //     Kokkos::View<double*, Kokkos::HostSpace> positions("positions", 3 *
    //     numberOfAtoms); Kokkos::View<double*, Kokkos::HostSpace>
    //     forces("forces", 3 * numberOfAtoms);

    //     // flatten positions
    //     auto flattenedPositions = simBox.flattenPositions();
    //     auto flattenedForces    = simBox.flattenForces();

    //     // copy flattened positions and forces to Kokkos View
    //     for (size_t i = 0; i < 3 * numberOfAtoms; ++i)
    //     {
    //         positions(i) = flattenedPositions[i];
    //         forces(i)    = flattenedForces[i];
    //     }

    //     // create Kokkos View on device
    //     Kokkos::View<double*, Kokkos::DefaultExecutionSpace>
    //     positionsDevice("positionsDevice", 3 * numberOfAtoms);
    //     Kokkos::View<double*, Kokkos::DefaultExecutionSpace>
    //     forcesDevice("forcesDevice", 3 * numberOfAtoms);

    //     // copy positions and forces to device
    //     Kokkos::deep_copy(positionsDevice, positions);
    //     Kokkos::deep_copy(forcesDevice, forces);

    //     Kokkos::parallel_reduce(numberOfAtoms, KOKKOS_LAMBDA(const size_t i,
    //     double &coulombEnergy, double &nonCoulombEnergy)
    //     {
    //         // calculate forces
    //         forcesDevice(i) = 0.0;

    //         // calculate coulombic energy
    //         coulombEnergy += 0.0;
    //         nonCoulombEnergy += 0.0;
    //     }, totalCoulombEnergy, totalNonCoulombEnergy);

    //     // copy forces back to host
    //     Kokkos::deep_copy(forces, forcesDevice);

    //     // copy forces back to simulation box
    //     for (size_t i = 0; i < 3 * numberOfAtoms; ++i)
    //     {
    //         flattenedForces[i] = forces(i);
    //     }

    //     // unflatten forces
    //     simBox.deFlattenForces(flattenedForces);

    //     // set total coulombic and non-coulombic energy
    //     physicalData.setCoulombEnergy(totalCoulombEnergy);
    //     physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    // }   // end of Kokkos scope
    // Kokkos::finalize();
}