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

using namespace simulationBox;

/**
 * @brief Constructor
 * 
 * @param numAtoms
 */
SimulationBoxCuda::SimulationBoxCuda(size_t numAtoms){
    // set number of atoms
    this->numAtoms = numAtoms;
    // allocate memory on device
    cudaMalloc((void **)&_atomTypes, numAtoms * sizeof(size_t));
    cudaMalloc((void **)&_molTypes, numAtoms * sizeof(size_t));
    cudaMalloc((void **)&_moleculeIndices, numAtoms * sizeof(size_t));
    cudaMalloc((void **)&_internatGlobalVDWTypes, numAtoms * sizeof(size_t));
    cudaMalloc((void **)&_positions, numAtoms * 3 * sizeof(double));
    cudaMalloc((void **)&_velocities, numAtoms * 3 * sizeof(double));
    cudaMalloc((void **)&_forces, numAtoms * 3 * sizeof(double));
    cudaMalloc((void **)&_shiftForeces, numAtoms * 3 * sizeof(double));
    cudaMalloc((void **)&_pratialCharges, numAtoms * sizeof(double));
    cudaMalloc((void **)&_masses, numAtoms * sizeof(double));
    cudaMalloc((void **)&_boxDimensions, 3 * sizeof(double));
}

/**
 * @brief Destructor
 */
SimulationBoxCuda::~SimulationBoxCuda(){
    // free memory on device
    cudaFree(_atomTypes);
    cudaFree(_molTypes);
    cudaFree(_moleculeIndices);
    cudaFree(_internatGlobalVDWTypes);
    cudaFree(_positions);
    cudaFree(_velocities);
    cudaFree(_forces);
    cudaFree(_shiftForeces);
    cudaFree(_pratialCharges);
    cudaFree(_masses);
    cudaFree(_boxDimensions);
}

/**
 * @brief Transfer data to device
 */
void SimulationBoxCuda::transferDataToDevice(SimulationBox &simulationBox){
    // transfer data to device
    transferAtomTypesFromSimulationBox(simulationBox);
    transferMolTypesFromSimulationBox(simulationBox);
    transferMoleculeIndicesFromSimulationBox(simulationBox);
    transferInternalGlobalVDWTypesFromSimulationBox(simulationBox);
    transferPositionsFromSimulationBox(simulationBox);
    transferVelocitiesFromSimulationBox(simulationBox);
    transferForcesFromSimulationBox(simulationBox);
    transferPartialChargesFromSimulationBox(simulationBox);
    transferMassesFromSimulationBox(simulationBox);
    transferBoxDimensionsFromSimulationBox(simulationBox);
}

/**
 * @brief Transfer atom types from simulation box
 */
void SimulationBoxCuda::transferAtomTypesFromSimulationBox(SimulationBox &simulationBox){
    // transfer atom types from simulation box
    cudaMemcpy(_atomTypes, simulationBox.getAtomTypes(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer mol types from simulation box
 */
void SimulationBoxCuda::transferMolTypesFromSimulationBox(SimulationBox &simulationBox){
    // transfer mol types from simulation box
    cudaMemcpy(_molTypes, simulationBox.getMolTypes(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer molecule indices from simulation box
 */
void SimulationBoxCuda::transferMoleculeIndicesFromSimulationBox(SimulationBox &simulationBox){
    // transfer molecule indices from simulation box
    cudaMemcpy(_moleculeIndices, simulationBox.getMoleculeIndices(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer internal global VDW types from simulation box
 */
void SimulationBoxCuda::transferInternalGlobalVDWTypesFromSimulationBox(SimulationBox &simulationBox){
    // transfer internal global VDW types from simulation box
    cudaMemcpy(_internatGlobalVDWTypes, simulationBox.getInternalGlobalVDWTypes(), numAtoms * sizeof(size_t), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer positions from simulation box
 */
void SimulationBoxCuda::transferPositionsFromSimulationBox(SimulationBox &simulationBox){
    // transfer positions from simulation box
    cudaMemcpy(_positions, simulationBox.getPositions(), numAtoms * 3 * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer velocities from simulation box
 */
void SimulationBoxCuda::transferVelocitiesFromSimulationBox(SimulationBox &simulationBox){
    // transfer velocities from simulation box
    cudaMemcpy(_velocities, simulationBox.getVelocities(), numAtoms * 3 * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer forces from simulation box
 */
void SimulationBoxCuda::transferForcesFromSimulationBox(SimulationBox &simulationBox){
    // transfer forces from simulation box
    cudaMemcpy(_forces, simulationBox.getForces(), numAtoms * 3 * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer partial charges from simulation box
 */
void SimulationBoxCuda::transferPartialChargesFromSimulationBox(SimulationBox &simulationBox){
    // transfer partial charges from simulation box
    cudaMemcpy(_pratialCharges, simulationBox.getPartialCharges(), numAtoms * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer masses from simulation box
 */
void SimulationBoxCuda::transferMassesFromSimulationBox(SimulationBox &simulationBox){
    // transfer masses from simulation box
    cudaMemcpy(_masses, simulationBox.getMasses(), numAtoms * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer box dimensions from simulation box
 */
void SimulationBoxCuda::transferBoxDimensionsFromSimulationBox(SimulationBox &simulationBox){
    // transfer box dimensions from simulation box
    cudaMemcpy(_boxDimensions, simulationBox.getBoxDimensions(), 3 * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Transfer positions to simulation box
 */
void SimulationBoxCuda::transferPositionsToSimulationBox(SimulationBox &simulationBox){
    // transfer positions to simulation box
    cudaMemcpy(simulationBox.getPositions(), _positions, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Transfer velocities to simulation box
 */
void SimulationBoxCuda::transferVelocitiesToSimulationBox(SimulationBox &simulationBox){
    // transfer velocities to simulation box
    cudaMemcpy(simulationBox.getVelocities(), _velocities, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Transfer forces to simulation box
 */
void SimulationBoxCuda::transferForcesToSimulationBox(SimulationBox &simulationBox){
    // transfer forces to simulation box
    cudaMemcpy(simulationBox.getForces(), _forces, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Transfer shift forces to simulation box
 */
void SimulationBoxCuda::transferShiftForcesToSimulationBox(SimulationBox &simulationBox){
    // transfer shift forces to simulation box
    cudaMemcpy(simulationBox.getShiftForces(), _shiftForeces, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Transfer data from device
 */
void SimulationBoxCuda::transferDataFromDevice(SimulationBox &simulationBox){
    // transfer data from device
    transferForcesFromDevice(simulationBox);
    transferShiftForcesFromDevice(simulationBox);
}

/**
 * @brief Transfer forces from device
 */
void SimulationBoxCuda::transferForcesFromDevice(SimulationBox &simulationBox){
    // transfer forces from device
    cudaMemcpy(_forces, simulationBox.getForces(), numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Transfer shift forces from device
 */
void SimulationBoxCuda::transferShiftForcesFromDevice(SimulationBox &simulationBox){
    // transfer shift forces from device
    cudaMemcpy(_shiftForeces, simulationBox.getShiftForces(), numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Get struct of simulation box cuda
 */
SimulationBoxCuda_t *SimulationBoxCuda::getSimulationBoxCuda(){
    // create simulation box cuda
    SimulationBoxCuda_t *simulationBoxCuda;

    // set simulation box cuda
    simulationBoxCuda->numAtoms = _numAtoms;
    simulationBoxCuda->numInternalGlobalVDWTypes = _numInternalGlobalVDWTypes;
    simulationBoxCuda->atomTypes = _atomTypes;
    simulationBoxCuda->molTypes = _molTypes;
    simulationBoxCuda->moleculeIndices = _moleculeIndices;
    simulationBoxCuda->internalGlobalVDWTypes = _internatGlobalVDWTypes;
    simulationBoxCuda->positions = _positions;
    simulationBoxCuda->velocities = _velocities;
    simulationBoxCuda->forces = _forces;
    simulationBoxCuda->shiftForeces = _shiftForeces;
    simulationBoxCuda->pratialCharges = _pratialCharges;
    simulationBoxCuda->masses = _masses;
    simulationBoxCuda->boxDimensions = _boxDimensions;

    // return simulation box cuda
    return simulationBoxCuda;
}