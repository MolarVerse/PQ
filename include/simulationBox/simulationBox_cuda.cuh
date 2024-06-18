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

#ifndef _SIMULATIONBOX_CUDA_HPP_
#define _SIMULATIONBOX_CUDA_HPP_

#include "cuda_runtime.h"
#include "simulationBox.hpp"
#include "timer.hpp"

/**
    * @brief Namespace for the simulation box
*/
namespace simulationBox
{

    /**
     * @brief Structure for the simulation box on the device
     */
    struct CudaSimulationBox_t
    {
        size_t numAtoms;
        size_t* atomTypes;
        size_t* molTypes;
        size_t* moleculeIndices;
        size_t* internalGlobalVDWTypes;
        double* positions;
        double* velocities;
        double* forces;
        double* shiftForces;
        double* partialCharges;
        double* masses;
        double* boxDimensions;
    }; // struct CudaSimulationBox_t

    /**
     * @brief Class for the simulation box on the device
     */
    class CudaSimulationBox : timings::Timer
    {
    private:
        // device variables
        size_t _numAtoms;
        size_t* _atomTypes;
        size_t* _molTypes;
        size_t* _moleculeIndices;
        size_t* _internalGlobalVDWTypes;
        double* _positions;
        double* _velocities;
        double* _forces;
        double* _shiftForces;
        double* _partialCharges;
        double* _masses;
        double* _boxDimensions;

    public:
        // constructor
        CudaSimulationBox(size_t numAtoms);

        // default destructor
        ~CudaSimulationBox();

        // transfer data to devide
        void transferDataToDevice(SimulationBox& simBox);
        // helper functions
        void transferAtomTypesFromSimulationBox(SimulationBox& simBox);
        void transferMolTypesFromSimulationBox(SimulationBox& simBox);
        void transferMoleculeIndicesFromSimulationBox(SimulationBox& simBox);
        void transferInternalGlobalVDWTypesFromSimulationBox(
            SimulationBox& simBox
        );
        void transferPositionsFromSimulationBox(SimulationBox& simBox);
        void transferVelocitiesFromSimulationBox(SimulationBox& simBox);
        void transferForcesFromSimulationBox(SimulationBox& simBox);
        void transferPartialChargesFromSimulationBox(SimulationBox& simBox);
        void transferMassesFromSimulationBox(SimulationBox& simBox);
        void transferBoxDimensionsFromSimulationBox(SimulationBox& simBox);
        void transferPositionsToSimulationBox(SimulationBox& simBox);
        void transferVelocitiesToSimulationBox(SimulationBox& simBox);
        void transferForcesToSimulationBox(SimulationBox& simBox);

        // transfer data from device
        void transferDataFromDevice(SimulationBox& simBox);
        // helper functions
        void transferForcesFromDevice(SimulationBox& simBox);
        void transferShiftForcesFromDevice(SimulationBox& simBox);

        // get device variables
        CudaSimulationBox_t* getSimulationBoxCuda();
    };

    // calculate shift vector
    __device__ __forceinline__ void calculateShiftVectorKernel(
        const double* positions,
        const double* boxDimensions,
        double* shiftForces
    );

};   // namespace simulationBox

#endif // SIMULATIONBOX_CUDA_HPP
