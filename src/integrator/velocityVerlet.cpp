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

#include "velocityVerlet.hpp"

#include "constants.hpp"
#include "simulationBox.hpp"
#include "simulationBox_API.hpp"
#include "timingsSettings.hpp"

using namespace integrator;
using namespace simulationBox;
using namespace settings;
using namespace constants;

VelocityVerlet::VelocityVerlet() : Integrator("VelocityVerlet"){};

/**
 * @brief applies first half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::firstStep(SimulationBox& simBox)
{
    startTimingsSection("Velocity Verlet - First Step");

    __DEBUG_INFO__("\n");
    __DEBUG_INFO__("Entering Velocity Verlet - First Step");
    __DEBUG_INFO__("\n");
    __POS_MIN_MAX_SUM_MEAN__(simBox);
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);

    const auto dt = TimingsSettings::getTimeStep();

    auto* const       velPtr    = simBox.getVelPtr();
    auto* const       posPtr    = simBox.getPosPtr();
    auto* const       forcesPtr = simBox.getForcesPtr();
    const auto* const massesPtr = simBox.getMassesPtr();
    const auto        nAtoms    = simBox.getNumberOfAtoms();

    const auto massFactor = dt * _V_VERLET_VELOCITY_FACTOR_;
    const auto posFactor  = dt * _FS_TO_S_;

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for                \
                is_device_ptr(velPtr, posPtr, massesPtr, forcesPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms * 3; ++i)
    {
        const size_t atomIndex = i / 3;

        velPtr[i]    += forcesPtr[i] / massesPtr[atomIndex] * massFactor;
        posPtr[i]    += velPtr[i] * posFactor;
        forcesPtr[i]  = 0.0;
    }

#ifdef __PQ_LEGACY__
    simBox.deFlattenVelocities();
    simBox.deFlattenPositions();
    simBox.deFlattenForces();
#endif

    calculateCenterOfMass(simBox);
    calculateCenterOfMassMolecules(simBox);

    __DEBUG_INFO__("\n");
    __DEBUG_INFO__("Exiting Velocity Verlet - First Step");
    __DEBUG_INFO__("\n");
    __POS_MIN_MAX_SUM_MEAN__(simBox);
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);

    stopTimingsSection("Velocity Verlet - First Step");
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox& simBox)
{
    startTimingsSection("Velocity Verlet - Second Step");
    __DEBUG_INFO__("\n");
    __DEBUG_INFO__("Entering Velocity Verlet - Second Step");
    __DEBUG_INFO__("\n");
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);

    const auto dt = TimingsSettings::getTimeStep();

    auto* const       velPtr    = simBox.getVelPtr();
    const auto* const massesPtr = simBox.getMassesPtr();
    const auto* const forcesPtr = simBox.getForcesPtr();
    const auto        nAtoms    = simBox.getNumberOfAtoms();

    const auto factor = dt * _V_VERLET_VELOCITY_FACTOR_;

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for                \
                is_device_ptr(velPtr, massesPtr, forcesPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms * 3; ++i)
    {
        const size_t atomIndex = i / 3;

        velPtr[i] += forcesPtr[i] / massesPtr[atomIndex] * factor;
    }

#ifdef __PQ_LEGACY__
    simBox.deFlattenVelocities();
#endif

    __DEBUG_INFO__("\n");
    __DEBUG_INFO__("Exiting Velocity Verlet - Second Step");
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __DEBUG_INFO__("\n");

    stopTimingsSection("Velocity Verlet - Second Step");
}