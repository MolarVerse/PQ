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
#include "orthorhombicBox.inl"
#include "simulationBox.hpp"
#include "simulationBox_API.hpp"
#include "timingsSettings.hpp"
#include "triclinicBox.inl"

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
    initFirstStep(simBox);

    const auto* const massesPtr = simBox.getMassesPtr();
    const auto* const boxParams = simBox.getBox().getBoxParamsPtr();

    auto* const velPtr    = simBox.getVelPtr();
    auto* const posPtr    = simBox.getPosPtr();
    auto* const forcesPtr = simBox.getForcesPtr();

    const auto nAtoms         = simBox.getNumberOfAtoms();
    const auto isOrthoRhombic = simBox.getBox().isOrthoRhombic();
    const auto dt             = TimingsSettings::getTimeStep();
    const auto massFactor     = dt * _V_VERLET_VELOCITY_FACTOR_;
    const auto posFactor      = dt * _FS_TO_S_;

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for                \
                is_device_ptr(velPtr, posPtr, massesPtr, forcesPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const size_t index  = i * 3 + j;
            velPtr[index]      += forcesPtr[index] / massesPtr[i] * massFactor;
            posPtr[index]      += velPtr[index] * posFactor;

            forcesPtr[index] = 0.0;
        }

        // TODO: make this more efficient
        if (isOrthoRhombic)
            imageOrthoRhombic(
                boxParams,
                posPtr[i * 3],
                posPtr[i * 3 + 1],
                posPtr[i * 3 + 2]
            );
        else
            imageTriclinic(
                boxParams,
                posPtr[i * 3],
                posPtr[i * 3 + 1],
                posPtr[i * 3 + 2]
            );
    }

    calculateCenterOfMass(simBox);
    calculateCenterOfMassMolecules(simBox);

    finalizeFirstStep(simBox);
}

/**
 * @brief initializes first step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::initFirstStep(SimulationBox& simBox)
{
    startTimingsSection(vvFirstStepMsg);

    __DEBUG_ENTER_FUNCTION__(vvFirstStepMsg);
    __POS_MIN_MAX_SUM_MEAN__(simBox);
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);

    __DEBUG_INFO__(std::format("Performing {}", vvFirstStepMsg));
}

/**
 * @brief finalizes first step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::finalizeFirstStep(SimulationBox& simBox)
{
#ifdef __PQ_LEGACY__
    simBox.deFlattenVelocities();
    simBox.deFlattenPositions();
    simBox.deFlattenForces();
#endif

    __POS_MIN_MAX_SUM_MEAN__(simBox);
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);
    __DEBUG_EXIT_FUNCTION__(vvFirstStepMsg);

    stopTimingsSection(vvFirstStepMsg);
}

/**
 * @brief applies second half step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::secondStep(SimulationBox& simBox)
{
    initSecondStep(simBox);

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

    finalizeSecondStep(simBox);
}

/**
 * @brief initializes second step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::initSecondStep(SimulationBox& simBox)
{
    startTimingsSection(vvSecondStepMsg);

    __DEBUG_ENTER_FUNCTION__(vvSecondStepMsg);
    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);

    __DEBUG_INFO__(std::format("Performing {}", vvSecondStepMsg));
}

/**
 * @brief finalizes second step of velocity verlet algorithm
 *
 * @param simBox
 */
void VelocityVerlet::finalizeSecondStep(SimulationBox& simBox)
{
#ifdef __PQ_LEGACY__
    simBox.deFlattenVelocities();
#endif

    __VEL_MIN_MAX_SUM_MEAN__(simBox);
    __FORCE_MIN_MAX_SUM_MEAN__(simBox);
    __DEBUG_EXIT_FUNCTION__(vvSecondStepMsg);

    stopTimingsSection(vvSecondStepMsg);
}