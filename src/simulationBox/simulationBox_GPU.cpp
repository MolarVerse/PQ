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
 * @brief Destructor for the SimulationBox class
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
 * @brief initialize device memory for simulationBox object
 *
 * @param device
 */
void SimulationBox::initDeviceMemory(device::Device& device)
{
    const size_t nAtoms     = getNumberOfAtoms();
    const size_t nMolecules = getNumberOfMolecules();
    const size_t size       = nAtoms * 3;

    assert(nAtoms > 0);
    assert(nMolecules > 0);

    device.deviceMalloc(&_posDevice, size);
    device.deviceMalloc(&_velDevice, size);
    device.deviceMalloc(&_forcesDevice, size);
    device.deviceMalloc(&_shiftForcesDevice, size);

    device.deviceMalloc(&_oldPosDevice, size);
    device.deviceMalloc(&_oldVelDevice, size);
    device.deviceMalloc(&_oldForcesDevice, size);

    device.deviceMalloc(&_massesDevice, nAtoms);
    device.deviceMalloc(&_chargesDevice, nAtoms);
    device.deviceMalloc(&_atomsPerMoleculeDevice, nMolecules);
    device.deviceMalloc(&_atomTypesDevice, nAtoms);
    device.deviceMalloc(&_molTypesDevice, nMolecules);
    device.deviceMalloc(&_internalGlobalVDWTypesDevice, nAtoms);

    device.checkErrors("SimulationBox device memory allocation");
}

/**
 * @brief copy position data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyPosTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_posDevice, _pos);
    device.checkErrors("SimulationBox copy position data to device");
}

/**
 * @brief copy velocity data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyVelTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_velDevice, _vel);
    device.checkErrors("SimulationBox copy velocity data to device");
}

/**
 * @brief copy forces data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyForcesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_forcesDevice, _forces);
    device.checkErrors("SimulationBox copy forces data to device");
}

/**
 * @brief copy shift forces data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyShiftForcesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_shiftForcesDevice, _shiftForces);
    device.checkErrors("SimulationBox copy shift forces data to device");
}

/**
 * @brief copy old position data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldPosTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_oldPosDevice, _pos);
    device.checkErrors("SimulationBox copy old position data to device");
}

/**
 * @brief copy old velocity data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldVelTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_oldVelDevice, _vel);
    device.checkErrors("SimulationBox copy old velocity data to device");
}

/**
 * @brief copy old forces data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldForcesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_oldForcesDevice, _forces);
    device.checkErrors("SimulationBox copy old forces data to device");
}

/**
 * @brief copy masses data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyMassesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_massesDevice, _masses);
    device.checkErrors("SimulationBox copy masses data to device");
}

/**
 * @brief copy charges data from host to device asynchronoysly
 *
 * @param device
 */
void SimulationBox::copyChargesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_chargesDevice, _charges);
    device.checkErrors("SimulationBox copy charges data to device");
}

/**
 * @brief copy number of atoms per molecule data from host to device
 * asynchronously
 *
 * @param device
 */
void SimulationBox::copyAtomsPerMoleculeTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_atomsPerMoleculeDevice, _atomsPerMolecule);
    device.checkErrors("SimulationBox copy atoms per molecule data to device");
}

/**
 * @brief copy the molecule indices data per atom from host to device
 * asynchronously
 *
 * @param device
 */
void SimulationBox::copyMoleculeIndicesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_moleculeIndicesDevice, _moleculeIndices);
    device.checkErrors("SimulationBox copy molecule indices data to device");
}

/**
 * @brief copy atomtypes data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyAtomTypesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_atomTypesDevice, _atomTypes);
    device.checkErrors("SimulationBox copy atom types data to device");
}

/**
 * @brief copy moltypes data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyMolTypesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(_molTypesDevice, _molTypes);
    device.checkErrors("SimulationBox copy mol types data to device");
}

/**
 * @brief copy internal global vdw types data from host to device asynchronously
 *
 * @param device
 */
void SimulationBox::copyInternalGlobalVDWTypesTo(device::Device& device)
{
    device.deviceMemcpyToAsync(
        _internalGlobalVDWTypesDevice,
        _internalGlobalVDWTypes
    );

    device.checkErrors(
        "SimulationBox copy internal global vdw types data to device"
    );
}

/**
 * @brief copy position data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyPosFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_pos, _posDevice);
    device.checkErrors("SimulationBox copy position data from device");
}

/**
 * @brief copy velocity data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyVelFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_vel, _velDevice);
    device.checkErrors("SimulationBox copy velocity data from device");
}

/**
 * @brief copy forces data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyForcesFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_forces, _forcesDevice);
    device.checkErrors("SimulationBox copy forces data from device");
}

/**
 * @brief copy old position data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldPosFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_pos, _oldPosDevice);
    device.checkErrors("SimulationBox copy old position data from device");
}

/**
 * @brief copy old velocity data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldVelFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_vel, _oldVelDevice);
    device.checkErrors("SimulationBox copy old velocity data from device");
}

/**
 * @brief copy old forces data from device to host asynchronously
 *
 * @param device
 */
void SimulationBox::copyOldForcesFrom(device::Device& device)
{
    device.deviceMemcpyFromAsync(_forces, _oldForcesDevice);
    device.checkErrors("SimulationBox copy old forces data from device");
}