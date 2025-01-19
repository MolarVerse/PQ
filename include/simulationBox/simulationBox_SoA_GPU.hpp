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

#ifndef __SIMULATION_BOX_SOA_GPU_HPP__
#define __SIMULATION_BOX_SOA_GPU_HPP__

#include "simulationBox_SoA.hpp"

namespace simulationBox
{
    class SimulationBoxSoAGPU : public SimulationBoxSoA
    {
       protected:
        std::shared_ptr<pq::Device> _device = nullptr;

        Real* _chargesDevice;
        Real* _massesDevice;
        Real* _molMassesDevice;

        size_t* _atomsPerMoleculeDevice;
        size_t* _moleculeIndicesDevice;
        size_t* _atomTypesDevice;
        size_t* _molTypesDevice;
        size_t* _internalGlobalVDWTypesDevice;
        size_t* _moleculeOffsetsDevice;

       public:
        ~SimulationBoxSoAGPU() override = default;

        void initDeviceMemorySimBoxSoA(
            device::Device& device,
            const size_t    nAtoms,
            const size_t    nMolecules
        );

        void copyChargesTo();
        void copyMassesTo();
        void copyMolMassesTo();

        void copyAtomsPerMoleculeTo();
        void copyMoleculeIndicesTo();
        void copyAtomTypesTo();
        void copyMolTypesTo();
        void copyInternalGlobalVDWTypesTo();
        void copyMoleculeOffsetsTo();

        void copyChargesFrom();
        void copyMassesFrom();
        void copyMolMassesFrom();

        void copyAtomsPerMoleculeFrom();
        void copyMoleculeIndicesFrom();
        void copyAtomTypesFrom();
        void copyMolTypesFrom();
        void copyInternalGlobalVDWTypesFrom();
        void copyMoleculeOffsetsFrom();

        [[nodiscard]] Real* getChargesPtr() override;
        [[nodiscard]] Real* getMassesPtr() override;
        [[nodiscard]] Real* getMolMassesPtr() override;

        [[nodiscard]] size_t* getAtomsPerMoleculePtr() override;
        [[nodiscard]] size_t* getMoleculeIndicesPtr() override;
        [[nodiscard]] size_t* getAtomTypesPtr() override;
        [[nodiscard]] size_t* getMolTypesPtr() override;
        [[nodiscard]] size_t* getInternalGlobalVDWTypesPtr() override;
        [[nodiscard]] size_t* getMoleculeOffsetsPtr() override;
    };
}   // namespace simulationBox

#endif   // __SIMULATION_BOX_SOA_GPU_HPP__