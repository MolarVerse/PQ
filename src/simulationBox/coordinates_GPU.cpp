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

#include "coordinates_GPU.hpp"

#include <cassert>
#include <cstddef>
#include <memory>

#include "device.hpp"   // IWYU pragma: keep
#include "settings.hpp"

using namespace simulationBox;
using namespace settings;

/**
 * @brief initialize device memory for simulationBoxGPU object
 *
 * @param device
 */
void CoordinatesGPU::initDeviceMemoryCoordinates(
    device::Device& device,
    const size_t    nAtoms,
    const size_t    nMolecules
)
{
    _device = std::make_shared<device::Device>(device);

    assert(nMolecules > 0);
    assert(nAtoms >= nMolecules);

    const auto atomSize = nAtoms * 3;
    const auto molSize  = nMolecules * 3;

    device.deviceMalloc(&_posDevice, atomSize);
    device.deviceMalloc(&_velDevice, atomSize);
    device.deviceMalloc(&_forcesDevice, atomSize);
    device.deviceMalloc(&_shiftForcesDevice, atomSize);

    device.deviceMalloc(&_oldPosDevice, atomSize);
    device.deviceMalloc(&_oldVelDevice, atomSize);
    device.deviceMalloc(&_oldForcesDevice, atomSize);

    device.deviceMalloc(&_comMoleculesDevice, molSize);

    device.checkErrors("Coordinates device memory allocation");
}

/**
 * @brief get pointer to positions
 *
 * @return pointer to positions
 */
Real* CoordinatesGPU::getPosPtr()
{
    if (Settings::useDevice())
        return _posDevice;
    else
        return _pos.data();
}

/**
 * @brief get pointer to velocities
 *
 * @return pointer to velocities
 */
Real* CoordinatesGPU::getVelPtr()
{
    if (Settings::useDevice())
        return _velDevice;
    else
        return _vel.data();
}

/**
 * @brief get pointer to forces
 *
 * @return pointer to forces
 */
Real* CoordinatesGPU::getForcesPtr()
{
    if (Settings::useDevice())
        return _forcesDevice;
    else
        return _forces.data();
}

/**
 * @brief get pointer to shift forces
 *
 * @return pointer to shift forces
 */
Real* CoordinatesGPU::getShiftForcesPtr()
{
    if (Settings::useDevice())
        return _shiftForcesDevice;
    else
        return _shiftForces.data();
}

/**
 * @brief get pointer to old positions
 *
 * @return pointer to old positions
 */
Real* CoordinatesGPU::getOldPosPtr()
{
    if (Settings::useDevice())
        return _oldPosDevice;
    else
        return _oldPos.data();
}

/**
 * @brief get pointer to old velocities
 *
 * @return pointer to old velocities
 */
Real* CoordinatesGPU::getOldVelPtr()
{
    if (Settings::useDevice())
        return _oldVelDevice;
    else
        return _oldVel.data();
}

/**
 * @brief get pointer to old forces
 *
 * @return pointer to old forces
 */
Real* CoordinatesGPU::getOldForcesPtr()
{
    if (Settings::useDevice())
        return _oldForcesDevice;
    else
        return _oldForces.data();
}

/**
 * @brief get pointer to center of mass of molecules
 *
 * @return pointer to center of mass of molecules
 */
Real* CoordinatesGPU::getComMoleculesPtr()
{
    if (Settings::useDevice())
        return _comMoleculesDevice;
    else
        return _comMolecules.data();
}

/**
 * @brief copy position data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyPosTo()
{
    _device->deviceMemcpyToAsync(_posDevice, _pos);
    _device->checkErrors("SimulationBox copy position data to device");
}

/**
 * @brief copy velocity data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyVelTo()
{
    _device->deviceMemcpyToAsync(_velDevice, _vel);
    _device->checkErrors("SimulationBox copy velocity data to device");
}

/**
 * @brief copy forces data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyForcesTo()
{
    _device->deviceMemcpyToAsync(_forcesDevice, _forces);
    _device->checkErrors("SimulationBox copy forces data to device");
}

/**
 * @brief copy shift forces data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyShiftForcesTo()
{
    _device->deviceMemcpyToAsync(_shiftForcesDevice, _shiftForces);
    _device->checkErrors("SimulationBox copy shift forces data to device");
}

/**
 * @brief copy old position data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldPosTo()
{
    _device->deviceMemcpyToAsync(_oldPosDevice, _oldPos);
    _device->checkErrors("SimulationBox copy old position data to device");
}

/**
 * @brief copy old velocity data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldVelTo()
{
    _device->deviceMemcpyToAsync(_oldVelDevice, _oldVel);
    _device->checkErrors("SimulationBox copy old velocity data to device");
}

/**
 * @brief copy old forces data from host to device asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldForcesTo()
{
    _device->deviceMemcpyToAsync(_oldForcesDevice, _oldForces);
    _device->checkErrors("SimulationBox copy old forces data to device");
}

/**
 * @brief copy center of mass of molecules data from host to device
 * asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyComMoleculesTo()
{
    _device->deviceMemcpyToAsync(_comMoleculesDevice, _comMolecules);
    _device->checkErrors(
        "SimulationBox copy center of mass of molecules data to device"
    );
}

/**
 * @brief copy position data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyPosFrom()
{
    _device->deviceMemcpyFromAsync(_pos, _posDevice);
    _device->checkErrors("SimulationBox copy position data from device");
}

/**
 * @brief copy velocity data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyVelFrom()
{
    _device->deviceMemcpyFromAsync(_vel, _velDevice);
    _device->checkErrors("SimulationBox copy velocity data from device");
}

/**
 * @brief copy forces data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyForcesFrom()
{
    _device->deviceMemcpyFromAsync(_forces, _forcesDevice);
    _device->checkErrors("SimulationBox copy forces data from device");
}

/**
 * @brief copy shift forces data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyShiftForcesFrom()
{
    _device->deviceMemcpyFromAsync(_shiftForces, _shiftForcesDevice);
    _device->checkErrors("SimulationBox copy shift forces data from device");
}

/**
 * @brief copy old position data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldPosFrom()
{
    _device->deviceMemcpyFromAsync(_oldPos, _oldPosDevice);
    _device->checkErrors("SimulationBox copy old position data from device");
}

/**
 * @brief copy old velocity data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldVelFrom()
{
    _device->deviceMemcpyFromAsync(_oldVel, _oldVelDevice);
    _device->checkErrors("SimulationBox copy old velocity data from device");
}

/**
 * @brief copy old forces data from device to host asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyOldForcesFrom()
{
    _device->deviceMemcpyFromAsync(_oldForces, _oldForcesDevice);
    _device->checkErrors("SimulationBox copy old forces data from device");
}

/**
 * @brief copy center of mass of molecules data from device to host
 * asynchronously
 *
 * @param device
 */
void CoordinatesGPU::copyComMoleculesFrom()
{
    _device->deviceMemcpyFromAsync(_comMolecules, _comMoleculesDevice);
    _device->checkErrors(
        "SimulationBox copy center of mass of molecules data from device"
    );
}