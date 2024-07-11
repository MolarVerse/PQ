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

#include "ringPolymerqmmdEngine.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity
#include <memory>       // for unique_ptr

#include "integrator.hpp"      // for Integrator
#include "manostat.hpp"        // for Manostat
#include "physicalData.hpp"    // for PhysicalData
#include "qmRunner.hpp"        // for QMRunner
#include "resetKinetics.hpp"   // for ResetKinetics
#include "staticMatrix.hpp"    // for StaticMatrix3x3
#include "thermostat.hpp"      // for Thermostat
#include "vector3d.hpp"        // for Vec3D

#ifdef WITH_MPI
#include <mpi.h>   // for MPI_Bcast, MPI_DOUBLE, MPI_COMM_WORLD

#include "mpi.hpp"   // for MPI
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

    std::ranges::for_each(
        _ringPolymerBeads,
        [this](auto &bead) { _integrator->firstStep(bead); }
    );

    qmCalculation();

    _constraints->applyDistanceConstraints(
        *_simulationBox,
        *_physicalData,
        calculateTotalSimulationTime()
    );

    coupleRingPolymerBeads();

    std::ranges::for_each(
        _ringPolymerBeads,
        [this](auto &bead)
        {
            _thermostat->applyThermostatOnForces(bead);

            _integrator->secondStep(bead);
        }
    );

    applyThermostat();

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _ringPolymerBeadsPhysicalData[i].calculateKinetics(_ringPolymerBeads[i]
        );

    applyManostat();

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _resetKinetics.reset(
            _step,
            _ringPolymerBeadsPhysicalData[i],
            _ringPolymerBeads[i]
        );

    combineBeads();

    _thermostat->applyTemperatureRamping();
}

/**
 * @brief qm calculation
 *
 * @details if mpi is activated, each process runs the qm calculation for a
 * single bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::qmCalculation()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _qmRunner->run(
                _ringPolymerBeads[i],
                _ringPolymerBeadsPhysicalData[i]
            );
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto forces   = _ringPolymerBeads[i].flattenForces();
        auto qmEnergy = _ringPolymerBeadsPhysicalData[i].getQMEnergy();
        auto virial =
            _ringPolymerBeadsPhysicalData[i].getVirial().toStdVector();

        ::MPI_Bcast(
            forces.data(),
            forces.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &qmEnergy,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            virial.data(),
            virial.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );

        _ringPolymerBeads[i].deFlattenForces(forces);
        _ringPolymerBeadsPhysicalData[i].setQMEnergy(qmEnergy);
        _ringPolymerBeadsPhysicalData[i].setVirial(
            linearAlgebra::StaticMatrix3x3(virial)
        );
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
 * @details if mpi is activated, each process runs the thermostat for a single
 * bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyThermostatHalfStep()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _thermostat->applyThermostatHalfStep(
                _ringPolymerBeads[i],
                _ringPolymerBeadsPhysicalData[i]
            );
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto velocities = _ringPolymerBeads[i].flattenVelocities();

        ::MPI_Bcast(
            velocities.data(),
            velocities.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );

        _ringPolymerBeads[i].deFlattenVelocities(velocities);
    }
}
#else
void RingPolymerQMMDEngine::applyThermostatHalfStep()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _thermostat->applyThermostatHalfStep(
            _ringPolymerBeads[i],
            _ringPolymerBeadsPhysicalData[i]
        );
}
#endif

/**
 * @brief apply thermostat
 *
 * @details if mpi is activated, each process runs the thermostat for a single
 * bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyThermostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _thermostat->applyThermostat(
                _ringPolymerBeads[i],
                _ringPolymerBeadsPhysicalData[i]
            );
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto velocities  = _ringPolymerBeads[i].flattenVelocities();
        auto temperature = _ringPolymerBeadsPhysicalData[i].getTemperature();
        auto noseHooverMomentumEnergy =
            _ringPolymerBeadsPhysicalData[i].getNoseHooverMomentumEnergy();
        auto noseHooverFrictionEnergy =
            _ringPolymerBeadsPhysicalData[i].getNoseHooverFrictionEnergy();

        ::MPI_Bcast(
            velocities.data(),
            velocities.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &temperature,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &noseHooverMomentumEnergy,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &noseHooverFrictionEnergy,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );

        _ringPolymerBeads[i].deFlattenVelocities(velocities);
        _ringPolymerBeadsPhysicalData[i].setTemperature(temperature);
        _ringPolymerBeadsPhysicalData[i].setNoseHooverMomentumEnergy(
            noseHooverMomentumEnergy
        );
        _ringPolymerBeadsPhysicalData[i].setNoseHooverFrictionEnergy(
            noseHooverFrictionEnergy
        );
    }
}
#else
void RingPolymerQMMDEngine::applyThermostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _thermostat->applyThermostat(
            _ringPolymerBeads[i],
            _ringPolymerBeadsPhysicalData[i]
        );
}
#endif

/**
 * @brief apply manostat
 *
 * @details if mpi is activated, each process runs the manostat for a single
 * bead or (portion of beads)
 *
 */
#ifdef WITH_MPI
void RingPolymerQMMDEngine::applyManostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            _manostat->applyManostat(
                _ringPolymerBeads[i],
                _ringPolymerBeadsPhysicalData[i]
            );
    }

    ::MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
    {
        auto positions  = _ringPolymerBeads[i].flattenPositions();
        auto velocities = _ringPolymerBeads[i].flattenVelocities();
        auto boxDimensions =
            _ringPolymerBeads[i].getBox().getBoxDimensions().toStdVector();
        auto volume  = _ringPolymerBeadsPhysicalData[i].getVolume();
        auto density = _ringPolymerBeadsPhysicalData[i].getDensity();

        ::MPI_Bcast(
            velocities.data(),
            velocities.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            positions.data(),
            positions.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            boxDimensions.data(),
            boxDimensions.size(),
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &volume,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );
        ::MPI_Bcast(
            &density,
            1,
            MPI_DOUBLE,
            i % mpi::MPI::getSize(),
            MPI_COMM_WORLD
        );

        _ringPolymerBeads[i].deFlattenVelocities(velocities);
        _ringPolymerBeads[i].deFlattenPositions(positions);
        _ringPolymerBeads[i].getBox().setBoxDimensions(
            {boxDimensions[0], boxDimensions[1], boxDimensions[2]}
        );
        _ringPolymerBeadsPhysicalData[i].setVolume(volume);
        _ringPolymerBeadsPhysicalData[i].setDensity(density);
    }
}
#else
void RingPolymerQMMDEngine::applyManostat()
{
    for (size_t i = 0; i < _ringPolymerBeads.size(); ++i)
        _manostat->applyManostat(
            _ringPolymerBeads[i],
            _ringPolymerBeadsPhysicalData[i]
        );
}
#endif