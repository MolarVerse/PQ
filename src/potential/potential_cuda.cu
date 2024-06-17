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
#include "simulationBox_cuda.cuh"

using namespace simulationBox;
using namespace potential;

/**
 * @brief Cuda kernel to calculate forces, coulombic and non-coulombic energy
 *
 * @param simulationBox
 * @param coulombEnergy
 * @param nonCoulombEnergy
 */
 __global__ void calculateForcesKernel(
    SimulationBoxCuda_t *simBox,
    double *coulombEnergy,
    double *nonCoulombEnergy
)
{
    // get thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread id is smaller than number of molecules
    if (i < simBox->numAtoms)
    {
        // forces
        double3 forces_i = {0.0, 0.0, 0.0};

        // shift forces
        double3 shiftForces_i = {0.0, 0.0, 0.0};

        // get atom type, molecule index, molecule type, partial charge and position
        size_t moleculeIndex_i = simBox->moleculeIndices[i];
        double partialCharge_i = simBox->partialCharges[i];
        size_t vDWType_i  = simBox->internalGlobalVDWTypes[i];
    }
}

/**
 * @brief calculates forces, coulombic and non-coulombic energy for CUDA
 * routine.
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialCuda::
    calculateForces(simulationBox::SimulationBox &simBox, 
        SimulationBoxCuda &simBoxCuda,
        physicalData::PhysicalData &physicalData)
{
    // start transfer timings -------------------------------------------------
    startTimingsSection("InterNonBonded - Transfer");

    // set total coulombic and non-coulombic energy
    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // cuda simulation box
    SimulationBoxCuda_t *simBox_struct = simBoxCuda.getSimulationBoxCuda();


    // end transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    // start calculation timings ----------------------------------------------
    startTimingsSection("InterNonBonded");

    size_t block_size = 256;
    size_t grid_size  = (simBox_struct->numAtoms + block_size - 1) / block_size;

    // calculate forces on device
    calculateForcesKernel<<<grid_size, block_size>>>(
        simBox_struct,
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

    // transfer data from device
    _simulationBoxCuda.transferDataFromDevice(simBox);

    // set total coulombic and non-coulombic energy
    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    // stop transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    return;
}
