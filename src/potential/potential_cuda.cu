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

#include "potential_cuda.cuh"   // for CudaPotential
#include "simulationBox_cuda.cuh"   // for CudaSimulationBox

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
    CudaSimulationBox_t* simBox,
    CudaLennardJones_t* lennardJones,
    CudaCoulombWolf_t* coulombWolf,
    double* coulombEnergy,
    double* nonCoulombEnergy
)
{
    // get thread id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread id is smaller than number of molecules
    if (i < simBox->numAtoms)
    {
        // coulombic energy
        double* coulombEnergy_i = &coulombEnergy[i];
        double* nonCoulombEnergy_i = &nonCoulombEnergy[i];

        // forces
        double3 forces_i = { 0.0, 0.0, 0.0 };

        // shift forces
        double3 shiftForces_i = { 0.0, 0.0, 0.0 };

        // get atom type, molecule index, molecule type, partial charge and position
        size_t moleculeIndex_i = simBox->moleculeIndices[i];
        double partialCharge_i = simBox->partialCharges[i];
        size_t vDWType_i = simBox->internalGlobalVDWTypes[i];

        for (size_t j = 0; j < simBox->numAtoms; ++j)
        {
            size_t moleculeIndex_j = simBox->moleculeIndices[j];

            if (moleculeIndex_i == moleculeIndex_j)
                continue;

            double dxyz[3] = {
                simBox->positions[i * 3 + 0] - simBox->positions[j * 3 + 0],
                simBox->positions[i * 3 + 1] - simBox->positions[j * 3 + 1],
                simBox->positions[i * 3 + 2] - simBox->positions[j * 3 + 2]
            };

            double txyz[3];

            calculateShiftVectorKernel(
                dxyz,
                simBox->boxDimensions,
                txyz
            );

            dxyz[0] += txyz[0];
            dxyz[1] += txyz[1];
            dxyz[2] += txyz[2];

            double distanceSquared =
                dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];

            if (distanceSquared > coulombWolf->coulombRadiusCutOff * coulombWolf->coulombRadiusCutOff)
                continue;

            double partialCharge_j = simBox->partialCharges[j];
            double distance = sqrt(distanceSquared);

            double force = 0.0;

            potential::calculateWolfKernel(
                coulombWolf,
                distance,
                partialCharge_i,
                partialCharge_j,
                force,
                coulombEnergy_i
            );

            size_t vdWType_j = simBox->internalGlobalVDWTypes[j];
            double nRCCutOff = lennardJones->radialCutoffs[vDWType_i * lennardJones->numAtomTypes + vdWType_j];

            if (distance < nRCCutOff)
            {
                potential::calculateLennardJonesKernel(
                    lennardJones, distance, force, vDWType_i, vdWType_j, nonCoulombEnergy_i
                );
            }

            force /= distance;

            double force_ij[3] = {
                force * dxyz[0],
                force * dxyz[1],
                force * dxyz[2]
            };

            simBox->shiftForces[i + 0] += force_ij[0] * txyz[0] / 2;
            simBox->shiftForces[i + 1] += force_ij[1] * txyz[1] / 2;
            simBox->shiftForces[i + 2] += force_ij[2] * txyz[2] / 2;

            simBox->forces[i + 0] += force_ij[0];
            simBox->forces[i + 1] += force_ij[1];
            simBox->forces[i + 2] += force_ij[2];
        }

    }
}

/**
 * @brief calculates forces, coulombic and non-coulombic energy for CUDA
 * routine.
 *
 * @param simBox
 * @param physicalData
 */
inline void CudaPotential::
calculateForces(SimulationBox& simBox,
    PhysicalData& physicalData,
    CudaSimulationBox& simBoxCuda,
    CudaLennardJones& lennardJonesCuda,
    CudaCoulombWolf& coulombWolfCuda
)
{
    // start transfer timings -------------------------------------------------
    startTimingsSection("InterNonBonded - Transfer");

    // set total coulombic and non-coulombic energy
    double totalCoulombEnergy = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // cuda simulation box
    CudaSimulationBox_t* simBox_struct = simBoxCuda.getSimulationBoxCuda();
    CudaLennardJones_t* lennardJones_struct = lennardJonesCuda.getCudaLennardJones();
    CudaCoulombWolf_t* coulombWolf_struct = coulombWolfCuda.getCudaCoulombWolf();

    // create energy arrays on device
    double* d_coulombEnergies;
    double* d_nonCoulombEnergies;

    cudaMallocManaged(&d_coulombEnergies, simBox.getNumberOfAtoms() * sizeof(double));
    cudaMallocManaged(&d_nonCoulombEnergies, simBox.getNumberOfAtoms() * sizeof(double));

    // end transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    // start calculation timings ----------------------------------------------
    startTimingsSection("InterNonBonded");

    size_t block_size = 256;
    size_t grid_size = (simBox_struct->numAtoms + block_size - 1) / block_size;

    // calculate forces on device
    calculateForcesKernel << <grid_size, block_size >> > (
        simBox_struct,
        lennardJones_struct,
        coulombWolf_struct,
        d_coulombEnergies,
        d_nonCoulombEnergies
        );

    // synchronize device
    cudaDeviceSynchronize();

    // copy data from device
    for (size_t i = 0; i < simBox.getNumberOfAtoms(); ++i)
    {
        totalCoulombEnergy += d_coulombEnergies[i];
        totalNonCoulombEnergy += d_nonCoulombEnergies[i];
    }

    // stop calculation timings ------------------------------------------------
    stopTimingsSection("InterNonBonded");

    // start transfer timings --------------------------------------------------
    startTimingsSection("InterNonBonded - Transfer");

    // half energy due to double counting
    totalCoulombEnergy *= 0.5;
    totalNonCoulombEnergy *= 0.5;

    // transfer data from device
    simBoxCuda.transferDataFromDevice(simBox);

    // set total coulombic and non-coulombic energy
    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    // stop transfer timings ---------------------------------------------------
    stopTimingsSection("InterNonBonded - Transfer");

    return;
}
