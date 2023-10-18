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
    auto beforeQMCalculation = [this](auto &bead)
    {
        _thermostat->applyThermostatHalfStep(_simulationBox, _physicalData);

        _integrator->firstStep(bead);
    };

    std::ranges::for_each(_ringPolymerBeads, beforeQMCalculation);

    qmCalculation();

    coupleRingPolymerBeads();

    std::ranges::for_each(_ringPolymerBeads,
                          [this](auto &bead)
                          {
                              _thermostat->applyThermostatOnForces(bead);

                              _integrator->secondStep(bead);
                          });

    applyThermostat();

    std::ranges::for_each(_ringPolymerBeads,
                          [this](auto &bead)
                          {
                              _physicalData.calculateKinetics(bead);

                              _manostat->applyManostat(bead, _physicalData);

                              _resetKinetics.reset(_step, _physicalData, bead);
                          });

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
    std::ranges::for_each(_ringPolymerBeads, [this](auto &bead) { _qmRunner->run(bead, _physicalData); });
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
            _thermostat->applyThermostat(_ringPolymerBeads[size_t(i)], _physicalData);
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
void RingPolymerQMMDEngine::applyThermostat(){
    std::ranges::for_each(_ringPolymerBeads, [this](auto &bead) { _thermostat->applyThermostat(bead, _physicalData); })};
#endif