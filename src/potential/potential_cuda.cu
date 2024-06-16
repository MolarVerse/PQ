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

#include "box.hpp"                // for Box
#include "coulombPotential.hpp"   // for CoulombPotential
#include "coulombWolf.hpp"        // for CoulombWolf
#include "cuda_runtime.h"
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potential.hpp"             // for Potential
#include "simulationBox.hpp"   // for SimulationBox

#include "potential.cuh"

namespace simulationBox
{
    class CellList;
}   // namespace simulationBox

using namespace potential;

/**
 * @brief Cuda kernel to calculate forces, coulombic and non-coulombic energy
 *
 * @param atomTypes
 * @param moleculeIndices
 * @param internalGlobalVDWTypes
 * @param molTypes
 * @param partialCharges
 * @param positions
 * @param forces
 * @param numberOfMolecules
 * @param coulombEnergy
 * @param nonCoulombEnergy
 */
 __global__ void calculateForcesKernel(
    double3 boxDimensions,
    size_t *atomTypes,
    size_t *moleculeIndices,
    size_t *internalGlobalVDWTypes,
    size_t *molTypes,
    double *partialCharges,
    double *positions,
    double *forces,
    size_t  numberOfMolecules,
    double coulombParameters[1],
    double *coulombEnergy,
    double *nonCoulombEnergy
)
{
    // get thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread id is smaller than number of molecules
    if (i < numberOfMolecules)
    {
        // forces
        double3 forces_i = {0.0, 0.0, 0.0};

        // shift forces
        double3 shiftForces_i = {0.0, 0.0, 0.0};

        // get atom type, molecule index, molecule type, partial charge and position
        size_t moleculeIndex = moleculeIndices[i];
        size_t atomType = atomTypes[i];
        size_t vDWType  = internalGlobalVDWTypes[i];

        // coulomb cutoff
        double coulombRadiusCutOff = coulombParameters[0];
        double coulR2 = coulombRadiusCutOff * coulombRadiusCutOff;

        // get positions of i-th molecule
        double3 position_i = {
            positions[3 * i],
            positions[3 * i + 1],
            positions[3 * i + 2]
        };

        for (size_t j = 0; j < numberOfMolecules; ++j)
        {
            const auto molecularIndex_j = moleculeIndices[j];

            // check if i-th molecule is equal to j-th molecule
            if (moleculeIndex == molecularIndex_j)
            {
                continue;
            }

            double3 position_j = {
                positions[3 * j],
                positions[3 * j + 1],
                positions[3 * j + 2]
            };

            // calculate direction vector
            // TODO: check if direction is correct
            double3 dxyz = {
                position_i.x - position_j.x,
                position_i.y - position_j.y,
                position_i.z - position_j.z
            };

            double3 txyz = calculateShiftVector(dxyz, boxDimensions);

            // shift positions
            dxyz.x += txyz.x;
            dxyz.y += txyz.y;
            dxyz.z += txyz.z;

            // distance squared 
            double r2 = dxyz.x * dxyz.x + dxyz.y * dxyz.y + dxyz.z * dxyz.z;

            // if distance squared is smaller than coulomb radius cutoff
            if (coulR2 < r2)
            {
                continue;
            }
        }
    }
}

/**
 * @brief calculate shift vector
 *
 * @param dxyz
 * @param box
 * @return double3
 */
__device__ double3 calculateShiftVector(double3 dxyz, double3 boxDimensions)
{
    double3 txyz;
    txyz.x = -boxDimensions.x * round(dxyz.x / boxDimensions.x);
    txyz.y = -boxDimensions.y * round(dxyz.y / boxDimensions.y);
    txyz.z = -boxDimensions.z * round(dxyz.z / boxDimensions.z);

    return txyz;
}

/**
 * @brief calculates coulombic
 * routine.
 *
 * @param distance
 * @param charge_i
 * @param charge_j
 * @param force_i
 * @return void
*/
__device__ void calculateCoulombic(
    double distance,
    double charge_i,
    double charge_j,
    double3 &force_i
)
{
    // calculate coulombic energy
    double coulombEnergy = charge_i * charge_j / distance;

    // calculate coulombic force
    double3 coulombForce = {
        coulombEnergy * distance,
        coulombEnergy * distance,
        coulombEnergy * distance
    };

    // add coulombic force to force_i
    force_i.x += coulombForce.x;
    force_i.y += coulombForce.y;
    force_i.z += coulombForce.z;
}


/**
 * @brief calculates forces, coulombic and non-coulombic energy for CUDA
 * routine.
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialCuda::
    calculateForces(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData, simulationBox::CellList &)
{
    // start transfer timings -------------------------------------------------
    startTimingsSection("InterNonBonded - Transfer");

    // set total coulombic and non-coulombic energy
    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // get box parameters
    const auto h_boxDimensions = simBox.getBox().getBoxDimensions();
    double3    d_boxDimensions = {
        h_boxDimensions[0],
        h_boxDimensions[1],
        h_boxDimensions[2]
    };

    
    // get Coulomb potential parameters
    const double coulombParameters = {
        CoulombPotential::getCoulombRadiusCutOff()

    };

    // get simulation parameters from simulation box
    const auto h_atomTypes       = simBox.flattenAtomTypes();
    const auto h_moleculeIndices = simBox.getMoleculeIndices();
    const auto h_internalGlobalVDWTypes =
        simBox.flattenInternalGlobalVDWTypes();
    const auto h_molTypes       = simBox.flattenMolTypes();
    const auto h_partialCharges = simBox.flattenPartialCharges();
    const auto h_positions      = simBox.flattenPositions();

    // TODO: check if forces are set to zero
    const auto h_forces = simBox.flattenForces();

    // initialize device memory
    size_t *d_atomTypes;
    size_t *d_moleculeIndices;
    size_t *d_internalGlobalVDWTypes;
    size_t *d_molTypes;
    double *d_partialCharges;
    double *d_positions;
    double *d_forces;

    // allocate memory on device
    cudaMallocManaged(&d_atomTypes, h_atomTypes.size() * sizeof(size_t));
    cudaMallocManaged(
        &d_moleculeIndices,
        h_moleculeIndices.size() * sizeof(size_t)
    );
    cudaMallocManaged(
        &d_internalGlobalVDWTypes,
        h_internalGlobalVDWTypes.size() * sizeof(size_t)
    );
    cudaMallocManaged(
        &d_internalGlobalVDWTypes,
        h_internalGlobalVDWTypes.size() * sizeof(size_t)
    );
    cudaMallocManaged(&d_molTypes, h_molTypes.size() * sizeof(size_t));
    cudaMallocManaged(
        &d_partialCharges,
        h_partialCharges.size() * sizeof(double)
    );
    cudaMallocManaged(&d_positions, h_positions.size() * sizeof(double));
    cudaMallocManaged(&d_forces, h_forces.size() * sizeof(double));

    // get number of atoms
    const size_t numberOfMolecules = simBox.getNumberOfMolecules();

    // end transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    // start calculation timings ----------------------------------------------
    startTimingsSection("InterNonBonded");

    size_t block_size = 256;
    size_t grid_size  = (numberOfMolecules + block_size - 1) / block_size;

    // calculate forces on device
    calculateForcesKernel<<<grid_size, block_size>>>(
        d_boxDimensions,
        d_atomTypes,
        d_moleculeIndices,
        d_internalGlobalVDWTypes,
        d_molTypes,
        d_partialCharges,
        d_positions,
        d_forces,
        coulombParameters,
        numberOfMolecules,
        &totalCoulombEnergy,
        &totalNonCoulombEnergy
    );

    // synchronize device
    cudaDeviceSynchronize();

    // stop calculation timings ------------------------------------------------
    stopTimingsSection("InterNonBonded");

    // start transfer timings --------------------------------------------------
    startTimingsSection("InterNonBonded - Transfer");

    // half energy due to double counting
    totalCoulombEnergy    *= 0.5;
    totalNonCoulombEnergy *= 0.5;

    simBox.deFlattenForces(h_forces);
    // TODO: check if shift forces transfer is needed

    // set total coulombic and non-coulombic energy
    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    // stop transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    // free memory on device
    cudaFree(d_atomTypes);
    cudaFree(d_moleculeIndices);
    cudaFree(d_internalGlobalVDWTypes);
    cudaFree(d_molTypes);
    cudaFree(d_partialCharges);
    cudaFree(d_positions);
    cudaFree(d_forces);

    return;
}
