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

#include "device.hpp"          // for device::Device
#include "deviceAPI.hpp"       // for device::Device
#include "simulationBox.hpp"   // header for the class definition

using namespace simulationBox;
using namespace device;

/**
 * @brief Destructor for the SimulationBoxGPU class
 */
SimulationBox::~SimulationBox()
{
    deviceFreeThrowError(_posDevice, "Freeing the position device memory");
    deviceFreeThrowError(_velDevice, "Freeing the velocity device memory");
    deviceFreeThrowError(_forcesDevice, "Freeing the forces device memory");
    deviceFreeThrowError(
        _shiftForcesDevice,
        "Freeing the shift forces device memory"
    );
    deviceFreeThrowError(
        _oldPosDevice,
        "Freeing the old position device memory"
    );
    deviceFreeThrowError(
        _oldVelDevice,
        "Freeing the old velocity device memory"
    );
    deviceFreeThrowError(
        _oldForcesDevice,
        "Freeing the old forces device memory"
    );
    deviceFreeThrowError(_massesDevice, "Freeing the masses device memory");
    deviceFreeThrowError(_chargesDevice, "Freeing the charges device memory");
    deviceFreeThrowError(
        _atomsPerMoleculeDevice,
        "Freeing the atoms per molecule device memory"
    );
    deviceFreeThrowError(
        _moleculeIndicesDevice,
        "Freeing the molecule indices device memory"
    );
    deviceFreeThrowError(
        _atomTypesDevice,
        "Freeing the atom types device memory"
    );
    deviceFreeThrowError(
        _molTypesDevice,
        "Freeing the mol types device memory"
    );
    deviceFreeThrowError(
        _internalGlobalVDWTypesDevice,
        "Freeing the internal global vdw types device memory"
    );
    _posDevice                    = nullptr;
    _velDevice                    = nullptr;
    _forcesDevice                 = nullptr;
    _shiftForcesDevice            = nullptr;
    _oldPosDevice                 = nullptr;
    _oldVelDevice                 = nullptr;
    _oldForcesDevice              = nullptr;
    _massesDevice                 = nullptr;
    _chargesDevice                = nullptr;
    _atomsPerMoleculeDevice       = nullptr;
    _moleculeIndicesDevice        = nullptr;
    _atomTypesDevice              = nullptr;
    _molTypesDevice               = nullptr;
    _internalGlobalVDWTypesDevice = nullptr;
}

/**
 * @brief initialize device memory for simulationBoxGPU object
 *
 * @param device
 */
void SimulationBox::initDeviceMemory(device::Device& device)
{
    initDeviceMemoryCoordinates(device, _nAtoms, _nMolecules);
    initDeviceMemorySimBoxSoA(device, _nAtoms, _nMolecules);

    device.checkErrors("SimulationBox device memory allocation");
}