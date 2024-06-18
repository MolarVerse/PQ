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

#include "cuda_runtime.h"
#include "simulationBox_cuda.cuh"
#include "simulationBox.hpp"
#include "vector3d.hpp"

using namespace simulationBox;

/**
 * @brief Constructor
 *
 * @param _numAtoms
 */
CudaSimulationBox::CudaSimulationBox(size_t numAtoms) {
    // set number of atoms
    _numAtoms = numAtoms;
    // allocate memory on device
    cudaMallocManaged((void**)&_atomTypes, _numAtoms * sizeof(size_t));
    cudaMallocManaged((void**)&_molTypes, _numAtoms * sizeof(size_t));
    cudaMallocManaged((void**)&_moleculeIndices, _numAtoms * sizeof(size_t));
    cudaMallocManaged((void**)&_internalGlobalVDWTypes, _numAtoms * sizeof(size_t));
    cudaMallocManaged((void**)&_positions, _numAtoms * 3 * sizeof(double));
    cudaMallocManaged((void**)&_velocities, _numAtoms * 3 * sizeof(double));
    cudaMallocManaged((void**)&_forces, _numAtoms * 3 * sizeof(double));
    cudaMallocManaged((void**)&_shiftForces, _numAtoms * 3 * sizeof(double));
    cudaMallocManaged((void**)&_partialCharges, _numAtoms * sizeof(double));
    cudaMallocManaged((void**)&_masses, _numAtoms * sizeof(double));
    cudaMallocManaged((void**)&_boxDimensions, 3 * sizeof(double));
}

/**
 * @brief Destructor
 */
CudaSimulationBox::~CudaSimulationBox() {
    // free memory on device
    cudaFree(_atomTypes);
    cudaFree(_molTypes);
    cudaFree(_moleculeIndices);
    cudaFree(_internalGlobalVDWTypes);
    cudaFree(_positions);
    cudaFree(_velocities);
    cudaFree(_forces);
    cudaFree(_shiftForces);
    cudaFree(_partialCharges);
    cudaFree(_masses);
    cudaFree(_boxDimensions);
}

/**
 * @brief Transfer data to device
 */
void CudaSimulationBox::transferDataToDevice(SimulationBox& simBox) {
    // transfer data to device
    transferAtomTypesFromSimulationBox(simBox);
    transferMolTypesFromSimulationBox(simBox);
    transferMoleculeIndicesFromSimulationBox(simBox);
    transferInternalGlobalVDWTypesFromSimulationBox(simBox);
    transferPositionsFromSimulationBox(simBox);
    transferVelocitiesFromSimulationBox(simBox);
    transferForcesFromSimulationBox(simBox);
    transferPartialChargesFromSimulationBox(simBox);
    transferMassesFromSimulationBox(simBox);
    transferBoxDimensionsFromSimulationBox(simBox);
}

/**
 * @brief Transfer atom types from simulation box
 */
void CudaSimulationBox::transferAtomTypesFromSimulationBox(SimulationBox& simBox) {
    // transfer atom types from simulation box
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _atomTypes[i] = simBox.getAtom(i).getAtomType();
    }
}

/**
 * @brief Transfer mol types from simulation box
 */
void CudaSimulationBox::transferMolTypesFromSimulationBox(SimulationBox& simBox) {
    // transfer mol types from simulation box
    auto molTypes = simBox.flattenMolTypes();
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _molTypes[i] = molTypes[i];
    }
}

/**
 * @brief Transfer molecule indices from simulation box
 */
void CudaSimulationBox::transferMoleculeIndicesFromSimulationBox(SimulationBox& simBox) {
    // transfer molecule indices from simulation box
    auto moleculeIndices = simBox.getMoleculeIndices();
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _moleculeIndices[i] = moleculeIndices[i];
    }
}

/**
 * @brief Transfer internal global VDW types from simulation box
 */
void CudaSimulationBox::transferInternalGlobalVDWTypesFromSimulationBox(SimulationBox& simBox) {
    // transfer internal global VDW types from simulation box
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _internalGlobalVDWTypes[i] = simBox.getAtom(i).getInternalGlobalVDWType();
    }
}

/**
 * @brief Transfer positions from simulation box
 */
void CudaSimulationBox::transferPositionsFromSimulationBox(SimulationBox& simBox) {
    // transfer positions from simulation box
    auto positions = simBox.flattenPositions();
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _positions[i * 3] = positions[i * 3];
        _positions[i * 3 + 1] = positions[i * 3 + 1];
        _positions[i * 3 + 2] = positions[i * 3 + 2];
    }
}

/**
 * @brief Transfer velocities from simulation box
 */
void CudaSimulationBox::transferVelocitiesFromSimulationBox(SimulationBox& simBox) {
    // transfer velocities from simulation box
    auto velocities = simBox.flattenVelocities();
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _velocities[i * 3] = velocities[i * 3];
        _velocities[i * 3 + 1] = velocities[i * 3 + 1];
        _velocities[i * 3 + 2] = velocities[i * 3 + 2];
    }
}

/**
 * @brief Transfer forces from simulation box
 */
void CudaSimulationBox::transferForcesFromSimulationBox(SimulationBox& simBox) {
    // transfer forces from simulation box
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _forces[i * 3] = 0;
        _forces[i * 3 + 1] = 0;
        _forces[i * 3 + 2] = 0;
    }
}

/**
 * @brief Transfer partial charges from simulation box
 */
void CudaSimulationBox::transferPartialChargesFromSimulationBox(SimulationBox& simBox) {
    // transfer partial charges from simulation box
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _partialCharges[i] = simBox.getAtom(i).getPartialCharge();
    }
}

/**
 * @brief Transfer masses from simulation box
 */
void CudaSimulationBox::transferMassesFromSimulationBox(SimulationBox& simBox) {
    // transfer masses from simulation box
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        _masses[i] = simBox.getAtom(i).getMass();
    }
}

/**
 * @brief Transfer box dimensions from simulation box
 */
void CudaSimulationBox::transferBoxDimensionsFromSimulationBox(SimulationBox& simBox) {
    // transfer box dimensions from simulation box
    auto boxDimensions = simBox.getBoxDimensions();
    for (size_t i = 0; i < 3; ++i)
    {
        _boxDimensions[i] = boxDimensions[i];
    }
}

/**
 * @brief Transfer data from device
 */
void CudaSimulationBox::transferDataFromDevice(SimulationBox& simBox) {
    // transfer data from device
    transferForcesFromDevice(simBox);
    transferShiftForcesFromDevice(simBox);
}

/**
 * @brief Transfer forces from device
 */
void CudaSimulationBox::transferForcesFromDevice(SimulationBox& simBox) {
    // transfer forces from device
    simBox.deFlattenForces(_forces);
}

/**
 * @brief Transfer shift forces to simulation box
 */
void CudaSimulationBox::transferShiftForcesFromDevice(SimulationBox& simBox) {
    // transfer shift forces to simulation box atoms
    for (size_t i = 0; i < _numAtoms; ++i)
    {
        linearAlgebra::Vec3D shiftForces;

        shiftForces[0] = _shiftForces[i * 3];
        shiftForces[1] = _shiftForces[i * 3 + 1];
        shiftForces[2] = _shiftForces[i * 3 + 2];

        simBox.getAtom(i).setShiftForce(shiftForces);
    }
}

/**
 * @brief Get struct of simulation box cuda
 */
CudaSimulationBox_t* CudaSimulationBox::getSimulationBoxCuda() {
    // create simulation box cuda
    CudaSimulationBox_t* simBoxCuda = new CudaSimulationBox_t;
    // set variables
    simBoxCuda->numAtoms = _numAtoms;
    simBoxCuda->atomTypes = _atomTypes;
    simBoxCuda->molTypes = _molTypes;
    simBoxCuda->moleculeIndices = _moleculeIndices;
    simBoxCuda->internalGlobalVDWTypes = _internalGlobalVDWTypes;
    simBoxCuda->positions = _positions;
    simBoxCuda->velocities = _velocities;
    simBoxCuda->forces = _forces;
    simBoxCuda->shiftForces = _shiftForces;
    simBoxCuda->partialCharges = _partialCharges;
    simBoxCuda->masses = _masses;
    simBoxCuda->boxDimensions = _boxDimensions;

    // return simulation box cuda
    return simBoxCuda;
}

/**
 * @brief Kernel to calculate shift vecotr
 */
__device__ __forceinline__ void calculateShiftVectorKernel(
    const double* positions,
    const double* boxDimensions,
    double* shiftForces
)
{
    shiftForces[0] =
        -boxDimensions[0] * nearbyint(positions[0] / boxDimensions[0]);
    shiftForces[1] =
        -boxDimensions[1] * nearbyint(positions[1] / boxDimensions[1]);
    shiftForces[2] =
        -boxDimensions[2] * nearbyint(positions[2] / boxDimensions[2]);
}