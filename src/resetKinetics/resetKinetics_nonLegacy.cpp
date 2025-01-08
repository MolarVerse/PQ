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

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <cstddef>      // for size_t
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _S_TO_FS_
#include "physicalData.hpp"                  // for PhysicalData
#include "resetKinetics.hpp"
#include "simulationBox.hpp"        // for SimulationBox
#include "staticMatrix.hpp"         // for operator*, operator+=
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for Vec3D, Vector3D, cross

using namespace resetKinetics;
using namespace linearAlgebra;
using namespace physicalData;
using namespace simulationBox;
using namespace constants;
using namespace settings;

/**
 * @brief reset the angular momentum of the system
 *
 * @details subtract angular momentum correction from all velocities -
 * correction is the total angular momentum divided by the total mass
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetAngularMomentum(SimulationBox &simBox)
{
    const auto centerOfMass = simBox.calculateCenterOfMass();

    _angularMomentum = simBox.calculateAngularMomentum(_momentum);

    StaticMatrix3x3 helperMatrix{0.0};

    auto helperMatrixXX = 0.0;
    auto helperMatrixXY = 0.0;
    auto helperMatrixXZ = 0.0;
    auto helperMatrixYY = 0.0;
    auto helperMatrixYZ = 0.0;
    auto helperMatrixZZ = 0.0;

    simBox.flattenPositions();
    simBox.flattenVelocities();

    const auto *const posPtr  = simBox.getPosPtr();
    const auto *const massPtr = simBox.getMassesPtr();
    auto *const       velPtr  = simBox.getVelPtr();

    const auto nAtoms = simBox.getNumberOfAtoms();
    // TODO: think of a nice way to use here tensorProduct

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for        \
                is_device_ptr(posPtr, velPtr, massPtr)      \
                map(centerOfMass)                           \
                reduction(+:helperMatrixXX, helperMatrixXY, \
                            helperMatrixXZ, helperMatrixYY, \
                            helperMatrixYZ, helperMatrixZZ)
#else
    #pragma omp parallel for                                \
                reduction(+:helperMatrixXX, helperMatrixXY, \
                            helperMatrixXZ, helperMatrixYY, \
                            helperMatrixYZ, helperMatrixZZ)
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto relPosX = posPtr[i * 3] - centerOfMass[0];
        const auto relPosY = posPtr[i * 3 + 1] - centerOfMass[1];
        const auto relPosZ = posPtr[i * 3 + 2] - centerOfMass[2];

        const auto mass = massPtr[i];

        helperMatrixXX += mass * relPosX * relPosX;
        helperMatrixXY += mass * relPosX * relPosY;
        helperMatrixXZ += mass * relPosX * relPosZ;
        helperMatrixYY += mass * relPosY * relPosY;
        helperMatrixYZ += mass * relPosY * relPosZ;
        helperMatrixZZ += mass * relPosZ * relPosZ;
    }

    helperMatrix[0][0] = helperMatrixXX;
    helperMatrix[0][1] = helperMatrixXY;
    helperMatrix[0][2] = helperMatrixXZ;
    helperMatrix[1][0] = helperMatrixXY;
    helperMatrix[1][1] = helperMatrixYY;
    helperMatrix[1][2] = helperMatrixYZ;
    helperMatrix[2][0] = helperMatrixXZ;
    helperMatrix[2][1] = helperMatrixYZ;
    helperMatrix[2][2] = helperMatrixZZ;

    const auto inertia = -helperMatrix + diagonalMatrix(trace(helperMatrix));
    const auto inverseInertia  = inverse(inertia);
    const auto angularVelocity = inverseInertia * _angularMomentum;

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(velPtr, massPtr)       \
                map(angularVelocity, centerOfMass)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto relPosX = posPtr[i * 3] - centerOfMass[0];
        const auto relPosY = posPtr[i * 3 + 1] - centerOfMass[1];
        const auto relPosZ = posPtr[i * 3 + 2] - centerOfMass[2];

        const auto relativePosition = Vec3D{relPosX, relPosY, relPosZ};

        const auto correction = cross(angularVelocity, relativePosition);

        velPtr[i * 3]     -= correction[0];
        velPtr[i * 3 + 1] -= correction[1];
        velPtr[i * 3 + 2] -= correction[2];
    }

    simBox.deFlattenVelocities();

    _temperature     = simBox.calculateTemperature();
    _momentum        = simBox.calculateMomentum();
    _angularMomentum = simBox.calculateAngularMomentum(_momentum);
}