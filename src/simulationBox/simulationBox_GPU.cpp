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

#include "../../include/device/deviceConfig.hpp"   // wrapper for HIP/CUDA runtime functions
#include "simulationBox.hpp"   // header for the class definition

using namespace simulationBox;
using namespace device;

/**
 * @brief initialize device memory for simulationBox object
 *
 */
void SimulationBox::initDeviceMemory()
{
    const size_t nAtoms = getNumberOfAtoms();
    const size_t size   = nAtoms * 3 * sizeof(Real);

    assert(nAtoms > 0);

    deviceMalloc((void**) &_posDevice, size);
    deviceMalloc((void**) &_velDevice, size);
    deviceMalloc((void**) &_forcesDevice, size);
}