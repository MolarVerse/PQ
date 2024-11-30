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
    _posDevice    = nullptr;
    _velDevice    = nullptr;
    _forcesDevice = nullptr;
}

/**
 * @brief initialize device memory for simulationBox object
 *
 * @param device
 */
void SimulationBox::initDeviceMemory(device::Device& device)
{
    const size_t nAtoms = getNumberOfAtoms();
    const size_t size   = nAtoms * 3;

    assert(nAtoms > 0);

    device.deviceMalloc(&_posDevice, size);
    device.deviceMalloc(&_velDevice, size);
    device.deviceMalloc(&_forcesDevice, size);
    device.deviceMalloc(&_massDevice, nAtoms);
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