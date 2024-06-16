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

#ifndef SIMULATIONBOX_CUDA_HPP
#define SIMULATIONBOX_CUDA_HPP

#include "cuda_runtime.h"
#include "simulationBox.hpp"


/**
    * @brief Namespace for the simulation box
*/
namespace simulationBox
{
    class SimulationBoxCuda : public SimulationBox
    {
        private:
            // device variables
            size_t *_atomTypes;
            size_t *_molTypes;
            size_t *_moleculeIndices;
            size_t *_internatGlobalVDWTypes;
            double *_positions;
            double *_velocities;
            double *_forces;
            double *_shiftForeces;
            double *_pratialCharges;
            double *_masses;
            double *_boxDimensions;

    public:
        // constructor
        explicit SimulationBoxCuda(size_t numAtoms){
            
        }

        // default constructor
        SimulationBoxCuda() = default;
        // default destructor
        ~SimulationBoxCuda() = default;

        void initCudaSimulationBox(simulationBox::SimulationBox &simBox);
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
        void transferShiftForcesToSimulationBox(SimulationBox& simBox);
    }



}   // namespace simulationBox

#endif // SIMULATIONBOX_CUDA_HPP
     