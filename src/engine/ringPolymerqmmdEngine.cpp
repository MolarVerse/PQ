/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "ringPolymerqmmdEngine.hpp"

#include "integrator.hpp"      // for Integrator
#include "manostat.hpp"        // for Manostat
#include "physicalData.hpp"    // for PhysicalData
#include "qmRunner.hpp"        // for QMRunner
#include "resetKinetics.hpp"   // for ResetKinetics
#include "thermostat.hpp"      // for Thermostat
#include "vector3d.hpp"        // for Vec3D

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity
#include <memory>       // for unique_ptr

#ifdef WITH_MPI
#include "mpi.hpp"   // for MPI

#include <mpi.h>   // for MPI_Bcast, MPI_DOUBLE, MPI_COMM_WORLD
#endif

using engine::RingPolymerQMMDEngine;

/**
 * @brief Takes one step in a ring polymer QM MD simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply thermostat half step
 * - Run QM calculations
 * - couple ring polymer beads
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void RingPolymerQMMDEngine::takeStep()
{

    applyThermostatHalfStep();

    std::ranges::for_each(_ringPolymerBeads, [this](auto &bead) { _integrator->firstStep(bead); });

    qmCalculation();

    coupleRingPolymerBeads();

    std::ranges::for_each(_ringPolymerBeads,
                          [this](auto &bead)
                          {
                              _thermostat->applyThermostatOnForces(bead);

                              _integrator->secondStep(bead);
                          });

    applyThermostat();

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _ringPolymerBeadsPhysicalData[i].calculateKinetics(_ringPolymerBeads[i]);

    applyManostat();

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _resetKinetics.reset(_step, _ringPolymerBeadsPhysicalData[i], _ringPolymerBeads[i]);

    combineBeads();
}

/**
 * @brief qm calculation
 *
 * @details if mpi is activated, each process runs the qm calculation for a single bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::qmCalculation()
{
    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _qmRunner->run(_ringPolymerBeads[size_t(i)], _physicalData);
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto forces = _ringPolymerBeads[size_t(i)].flattenForces();

        ::MPI_Bcast(forces.data(), forces.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);

        _ringPolymerBeads[size_t(i)].deFlattenForces(forces);
    }
}
#else
void RingPolymerQMMDEngine::qmCalculation()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _qmRunner->run(_ringPolymerBeads[i], _ringPolymerBeadsPhysicalData[i]);
}
#endif

/**
 * @brief apply thermostat for half step
 *
 * @details if mpi is activated, each process runs the thermostat for a single bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyThermostatHalfStep()
{
    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _thermostat->applyThermostatHalfStep(_ringPolymerBeads[size_t(i)], _ringPolymerBeadsPhysicalData[size_t(i)]);
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto velocities = _ringPolymerBeads[size_t(i)].flattenVelocities();

        ::MPI_Bcast(velocities.data(), velocities.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);

        _ringPolymerBeads[size_t(i)].deFlattenVelocities(velocities);
    }
}
#else
void RingPolymerQMMDEngine::applyThermostatHalfStep()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _thermostat->applyThermostatHalfStep(_ringPolymerBeads[i], _ringPolymerBeadsPhysicalData[i]);
}
#endif

/**
 * @brief apply thermostat
 *
 * @details if mpi is activated, each process runs the thermostat for a single bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyThermostat()
{
    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _thermostat->applyThermostat(_ringPolymerBeads[size_t(i)], _ringPolymerBeadsPhysicalData[size_t(i)]);
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto velocities = _ringPolymerBeads[size_t(i)].flattenVelocities();

        ::MPI_Bcast(velocities.data(), velocities.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);

        _ringPolymerBeads[size_t(i)].deFlattenVelocities(velocities);
    }
}
#else
void RingPolymerQMMDEngine::applyThermostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _thermostat->applyThermostat(_ringPolymerBeads[i], _ringPolymerBeadsPhysicalData[i]);
}
#endif

/**
 * @brief apply manostat
 *
 * @details if mpi is activated, each process runs the manostat for a single bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyManostat()
{
    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _manostat->applyManostat(_ringPolymerBeads[size_t(i)], _ringPolymerBeadsPhysicalData[size_t(i)]);
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto positions     = _ringPolymerBeads[size_t(i)].flattenPositions();
        auto velocities    = _ringPolymerBeads[size_t(i)].flattenVelocities();
        auto boxDimensions = _ringPolymerBeads[size_t(i)].getBox().getBoxDimensions().toStdVector();

        ::MPI_Bcast(velocities.data(), velocities.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);
        ::MPI_Bcast(positions.data(), positions.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);
        ::MPI_Bcast(boxDimensions.data(), boxDimensions.size(), MPI_DOUBLE, i % mpi::MPI::getSize(), MPI_COMM_WORLD);

        _ringPolymerBeads[size_t(i)].deFlattenVelocities(velocities);
        _ringPolymerBeads[size_t(i)].deFlattenPositions(positions);
        _ringPolymerBeads[size_t(i)].getBox().setBoxDimensions({boxDimensions[0], boxDimensions[1], boxDimensions[2]});
    }
}
#else
void RingPolymerQMMDEngine::applyManostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _manostat->applyManostat(_ringPolymerBeads[i], _ringPolymerBeadsPhysicalData[i]);
}
#endif