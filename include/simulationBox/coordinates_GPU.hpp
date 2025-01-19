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

#ifndef __COORDINATES_GPU_HPP__
#define __COORDINATES_GPU_HPP__

#include <cstddef>
#include <memory>

#include "coordinates.hpp"
#include "typeAliases.hpp"

namespace simulationBox
{
    class CoordinatesGPU : public Coordinates
    {
       protected:
        std::shared_ptr<pq::Device> _device = nullptr;

        Real* _posDevice;
        Real* _velDevice;
        Real* _forcesDevice;
        Real* _shiftForcesDevice;
        Real* _oldPosDevice;
        Real* _oldVelDevice;
        Real* _oldForcesDevice;

        Real* _comMoleculesDevice;

       public:
        void initDeviceMemoryCoordinates(
            device::Device& device,
            const size_t    nAtoms,
            const size_t    nMolecules
        );

        void copyPosTo();
        void copyVelTo();
        void copyForcesTo();
        void copyShiftForcesTo();
        void copyOldPosTo();
        void copyOldVelTo();
        void copyOldForcesTo();
        void copyComMoleculesTo();

        void copyPosFrom();
        void copyVelFrom();
        void copyForcesFrom();
        void copyShiftForcesFrom();
        void copyOldPosFrom();
        void copyOldVelFrom();
        void copyOldForcesFrom();
        void copyComMoleculesFrom();

        [[nodiscard]] Real* getPosPtr() override;
        [[nodiscard]] Real* getVelPtr() override;
        [[nodiscard]] Real* getForcesPtr() override;
        [[nodiscard]] Real* getShiftForcesPtr() override;
        [[nodiscard]] Real* getOldPosPtr() override;
        [[nodiscard]] Real* getOldVelPtr() override;
        [[nodiscard]] Real* getOldForcesPtr() override;
        [[nodiscard]] Real* getComMoleculesPtr() override;
    };

}   // namespace simulationBox

#endif   // __COORDINATES_GPU_HPP__