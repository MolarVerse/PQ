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

#include "simulationBox_SoA_GPU.hpp"

#include <cassert>
#include <cstddef>
#include <memory>

#include "device.hpp"
#include "settings.hpp"
#include "typeAliases.hpp"

using namespace simulationBox;
using namespace settings;

/**
 * @brief initialize device memory
 *
 * @param device
 * @param nAtoms
 * @param nMolecules
 */
void SimulationBoxSoAGPU::initDeviceMemorySimBoxSoA(
    device::Device& device,
    const size_t    nAtoms,
    const size_t    nMolecules
)
{
    _device = std::make_shared<device::Device>(device);

    assert(nMolecules > 0);
    assert(nAtoms >= nMolecules);

    device.deviceMalloc(&_chargesDevice, nAtoms);
    device.deviceMalloc(&_massesDevice, nAtoms);
    device.deviceMalloc(&_molMassesDevice, nMolecules);

    device.deviceMalloc(&_moleculeIndicesDevice, nAtoms);
    device.deviceMalloc(&_atomTypesDevice, nAtoms);
    device.deviceMalloc(&_internalGlobalVDWTypesDevice, nAtoms);
    device.deviceMalloc(&_atomsPerMoleculeDevice, nMolecules);
    device.deviceMalloc(&_molTypesDevice, nMolecules);
    device.deviceMalloc(&_moleculeOffsetsDevice, nMolecules);

    device.checkErrors("SimulationBoxSoAGPU device memory allocation");
}

/**
 * @brief get charges pointer
 *
 * @return pointer to charges
 */
Real* SimulationBoxSoAGPU::getChargesPtr()
{
    if (Settings::useDevice())
        return _chargesDevice;
    else
        return _charges.data();
}

/**
 * @brief get masses pointer
 *
 * @return pointer to masses
 */
Real* SimulationBoxSoAGPU::getMassesPtr()
{
    if (Settings::useDevice())
        return _massesDevice;
    else
        return _masses.data();
}

/**
 * @brief get mol masses pointer
 *
 * @return pointer to mol masses
 */
Real* SimulationBoxSoAGPU::getMolMassesPtr()
{
    if (Settings::useDevice())
        return _molMassesDevice;
    else
        return _molMasses.data();
}

/**
 * @brief get atoms per molecule pointer
 *
 * @return pointer to atoms per molecule
 */
size_t* SimulationBoxSoAGPU::getAtomsPerMoleculePtr()
{
    if (Settings::useDevice())
        return _atomsPerMoleculeDevice;
    else
        return _atomsPerMolecule.data();
}

/**
 * @brief get atom types pointer
 *
 * @return pointer to atom types
 */
size_t* SimulationBoxSoAGPU::getAtomTypesPtr()
{
    if (Settings::useDevice())
        return _atomTypesDevice;
    else
        return _atomTypes.data();
}

/**
 * @brief get internal global VDW types pointer
 *
 * @return pointer to internal global VDW types
 */
size_t* SimulationBoxSoAGPU::getInternalGlobalVDWTypesPtr()
{
    if (Settings::useDevice())
        return _internalGlobalVDWTypesDevice;
    else
        return _internalGlobalVDWTypes.data();
}

/**
 * @brief get molecule indices pointer
 *
 * @return pointer to molecule indices
 */
size_t* SimulationBoxSoAGPU::getMoleculeIndicesPtr()
{
    if (Settings::useDevice())
        return _moleculeIndicesDevice;
    else
        return _moleculeIndices.data();
}

/**
 * @brief get molecule types pointer
 *
 * @return pointer to molecule types
 */
size_t* SimulationBoxSoAGPU::getMolTypesPtr()
{
    if (Settings::useDevice())
        return _molTypesDevice;
    else
        return _molTypes.data();
}

/**
 * @brief get molecule offsets pointer
 *
 * @return pointer to molecule offsets
 */
size_t* SimulationBoxSoAGPU::getMoleculeOffsetsPtr()
{
    if (Settings::useDevice())
        return _moleculeOffsetsDevice;
    else
        return _moleculeOffsets.data();
}

/**
 * @brief copy charges data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyChargesTo()
{
    _device->deviceMemcpyToAsync(_chargesDevice, _charges);
    _device->checkErrors("SimulationBoxSoAGPU copy charges data to device");
}

/**
 * @brief copy masses data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMassesTo()
{
    _device->deviceMemcpyToAsync(_massesDevice, _masses);
    _device->checkErrors("SimulationBoxSoAGPU copy masses data to device");
}

/**
 * @brief copy mol masses data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMolMassesTo()
{
    _device->deviceMemcpyToAsync(_molMassesDevice, _molMasses);
    _device->checkErrors("SimulationBoxSoAGPU copy mol masses data to device");
}

/**
 * @brief copy atoms per molecule data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyAtomsPerMoleculeTo()
{
    _device->deviceMemcpyToAsync(_atomsPerMoleculeDevice, _atomsPerMolecule);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy atoms per molecule data to device"
    );
}

/**
 * @brief copy the molecule indices data per atom from host to device
 * asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMoleculeIndicesTo()
{
    _device->deviceMemcpyToAsync(_moleculeIndicesDevice, _moleculeIndices);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy molecule indices data to device"
    );
}

/**
 * @brief copy atom types data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyAtomTypesTo()
{
    _device->deviceMemcpyToAsync(_atomTypesDevice, _atomTypes);
    _device->checkErrors("SimulationBoxSoAGPU copy atom types data to device");
}

/**
 * @brief copy mol types data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMolTypesTo()
{
    _device->deviceMemcpyToAsync(_molTypesDevice, _molTypes);
    _device->checkErrors("SimulationBoxSoAGPU copy mol types data to device");
}

/**
 * @brief copy internal global vdw types data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyInternalGlobalVDWTypesTo()
{
    _device->deviceMemcpyToAsync(
        _internalGlobalVDWTypesDevice,
        _internalGlobalVDWTypes
    );
    _device->checkErrors(
        "SimulationBoxSoAGPU copy internal global vdw types data to device"
    );
}

/**
 * @brief copy molecule offsets data from host to device asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMoleculeOffsetsTo()
{
    _device->deviceMemcpyToAsync(_moleculeOffsetsDevice, _moleculeOffsets);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy molecule offsets data to device"
    );
}

/**
 * @brief copy charges data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyChargesFrom()
{
    _device->deviceMemcpyFromAsync(_charges, _chargesDevice);
    _device->checkErrors("SimulationBoxSoAGPU copy charges data from device");
}

/**
 * @brief copy masses data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMassesFrom()
{
    _device->deviceMemcpyFromAsync(_masses, _massesDevice);
    _device->checkErrors("SimulationBoxSoAGPU copy masses data from device");
}

/**
 * @brief copy mol masses data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMolMassesFrom()
{
    _device->deviceMemcpyFromAsync(_molMasses, _molMassesDevice);
    _device->checkErrors("SimulationBoxSoAGPU copy mol masses data from device"
    );
}

/**
 * @brief copy atoms per molecule data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyAtomsPerMoleculeFrom()
{
    _device->deviceMemcpyFromAsync(_atomsPerMolecule, _atomsPerMoleculeDevice);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy atoms per molecule data from device"
    );
}

/**
 * @brief copy molecule indices data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMoleculeIndicesFrom()
{
    _device->deviceMemcpyFromAsync(_moleculeIndices, _moleculeIndicesDevice);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy molecule indices data from device"
    );
}

/**
 * @brief copy atom types data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyAtomTypesFrom()
{
    _device->deviceMemcpyFromAsync(_atomTypes, _atomTypesDevice);
    _device->checkErrors("SimulationBoxSoAGPU copy atom types data from device"
    );
}

/**
 * @brief copy mol types data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMolTypesFrom()
{
    _device->deviceMemcpyFromAsync(_molTypes, _molTypesDevice);
    _device->checkErrors("SimulationBoxSoAGPU copy mol types data from device");
}

/**
 * @brief copy internal global vdw types data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyInternalGlobalVDWTypesFrom()
{
    _device->deviceMemcpyFromAsync(
        _internalGlobalVDWTypes,
        _internalGlobalVDWTypesDevice
    );
    _device->checkErrors(
        "SimulationBoxSoAGPU copy internal global vdw types data from device"
    );
}

/**
 * @brief copy molecule offsets data from device to host asynchronously
 *
 */
void SimulationBoxSoAGPU::copyMoleculeOffsetsFrom()
{
    _device->deviceMemcpyFromAsync(_moleculeOffsets, _moleculeOffsetsDevice);
    _device->checkErrors(
        "SimulationBoxSoAGPU copy molecule offsets data from device"
    );
}