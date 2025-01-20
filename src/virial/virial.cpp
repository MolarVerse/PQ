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

#include "virial.hpp"

#include <cstddef>   // for size_t

#include "debug.hpp"
#include "linearAlgebra.hpp"   // for tensor3D, tensorProduct, atomicAdd
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData, physicalData, simulationBox
#include "simulationBox.hpp"   // for SimulationBox

using namespace virial;
using namespace simulationBox;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief calculate virial for general systems
 *
 * @details It calculates the virial for all atoms in the simulation box without
 * any corrections. It already sets the virial in the physicalData object
 *
 * @param simBox
 * @param data
 */
void Virial::calculateVirial(SimulationBox &simBox, PhysicalData &data)
{
    startTimingsSection("Virial");

    __DEBUG_ENTER_FUNCTION__("Virial");

    Real virial[9] = {0.0};

    const auto        nAtoms         = simBox.getNumberOfAtoms();
    const auto *const forcesPtr      = simBox.getForcesPtr();
    auto *const       shiftForcesPtr = simBox.getShiftForcesPtr();
    const auto *const posPtr         = simBox.getPosPtr();

#ifdef __PQ_GPU__
    // clang-format off
    #pragma omp target teams distribute parallel for          \
                is_device_ptr(forcesPtr, posPtr)              \
                map(virial)
#else
    #pragma omp parallel for
    // clang-format on
#endif
    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto posX   = posPtr[i * 3];
        const auto posY   = posPtr[i * 3 + 1];
        const auto posZ   = posPtr[i * 3 + 2];
        const auto forceX = forcesPtr[i * 3];
        const auto forceY = forcesPtr[i * 3 + 1];
        const auto forceZ = forcesPtr[i * 3 + 2];

        Real help[9] = {0.0};
        tensorProduct(help, posX, posY, posZ, forceX, forceY, forceZ);

        for (size_t k = 0; k < 3; ++k)
        {
            help[k * 3 + k]           += shiftForcesPtr[i * 3 + k];
            shiftForcesPtr[i * 3 + k]  = 0.0;
        }

        for (size_t k = 0; k < 9; ++k)
            linearAlgebra::atomicAdd(&virial[k], help[k]);
    }

    _virial = tensor3D(virial);
    data.setVirial(_virial);

#ifdef __PQ_LEGACY__
    simBox.deFlattenShiftForces();
#endif

    __DEBUG_VIRIAL__(_virial);
    __DEBUG_EXIT_FUNCTION__("Virial");

    stopTimingsSection("Virial");
}

/**
 * @brief set the virial
 *
 * @param virial
 */
void Virial::setVirial(const tensor3D &virial) { _virial = virial; }

/**
 * @brief get the virial
 *
 * @return tensor3D
 */
tensor3D Virial::getVirial() const { return _virial; }

/**
 * @brief get the virial type
 *
 * @return std::string
 */
std::string Virial::getVirialType() const { return _virialType; }