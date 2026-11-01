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

#include <algorithm>

#include "integrator.hpp"
#include "manostat.hpp"
#include "thermostat.hpp"

#ifdef WITH_MPI
#include <mpi.h>

#include "mpi.hpp"
#endif

namespace engine
{

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
            getRingPolymerBeads(),
            [this](auto &bead) { getIntegrator().firstStep(bead); }
        );

        qmCalculation();

        getConstraints()->applyDistanceConstraints(
            getSimulationBox(),
            getPhysicalData(),
            calculateTotalSimulationTime()
        );

        coupleRingPolymerBeads();

        std::ranges::for_each(
            getRingPolymerBeads(),
            [this](auto &bead)
            {
                getThermostat().applyThermostatOnForces(bead);

                getIntegrator().secondStep(bead);
            }
        );

        applyThermostat();

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &bead = getRingPolymerBeads()[i];
            getRingPolymerBeadsPhysicalData()[i].calculateKinetics(bead);
        }

        applyManostat();

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &data = getRingPolymerBeadsPhysicalData()[i];
            auto &bead = getRingPolymerBeads()[i];

            getResetKinetics().reset(getStep(), data, bead);
        }

        combineBeads();

        getThermostat().applyTemperatureRamping();
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
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            {
                auto &bead = getRingPolymerBeads()[i];
                auto &data = getRingPolymerBeadsPhysicalData()[i];

                getQMRunner().run(bead, data);
            }
        }

        ::MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto forces   = getRingPolymerBeads()[i].flattenForces();
            auto qmEnergy = getRingPolymerBeadsPhysicalData()[i].getQMEnergy();

            auto virial =
                getRingPolymerBeadsPhysicalData()[i].getVirial().toStdVector();

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

            getRingPolymerBeads()[i].deFlattenForces(forces);
            getRingPolymerBeadsPhysicalData()[i].setQMEnergy(qmEnergy);
            getRingPolymerBeadsPhysicalData()[i].setVirial(
                linalg::StaticMatrix3x3(virial)
            );
        }
    }
#else
    void RingPolymerQMMDEngine::qmCalculation()
    {
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            getQMRunner().run(
                getRingPolymerBeads()[i],
                getRingPolymerBeadsPhysicalData()[i]
            );
        }
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
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            {
                auto &bead = getRingPolymerBeads()[i];
                auto &data = getRingPolymerBeadsPhysicalData()[i];

                getThermostat().applyThermostatHalfStep(bead, data);
            }
        }

        ::MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto velocities = getRingPolymerBeads()[i].flattenVelocities();

            ::MPI_Bcast(
                velocities.data(),
                velocities.size(),
                MPI_DOUBLE,
                i % mpi::MPI::getSize(),
                MPI_COMM_WORLD
            );

            getRingPolymerBeads()[i].deFlattenVelocities(velocities);
        }
    }
#else
    void RingPolymerQMMDEngine::applyThermostatHalfStep()
    {
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &bead = getRingPolymerBeads()[i];
            auto &data = getRingPolymerBeadsPhysicalData()[i];

            getThermostat().applyThermostatHalfStep(bead, data);
        }
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
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            {
                auto &bead = getRingPolymerBeads()[i];
                auto &data = getRingPolymerBeadsPhysicalData()[i];

                getThermostat().applyThermostat(bead, data);
            }
        }

        ::MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &data = getRingPolymerBeadsPhysicalData()[i];

            auto velocities  = getRingPolymerBeads()[i].flattenVelocities();
            auto temperature = data.getTemperature();

            auto noseHooverMomentumEnergy = data.getNoseHooverMomentumEnergy();
            auto noseHooverFrictionEnergy = data.getNoseHooverFrictionEnergy();

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

            getRingPolymerBeads()[i].deFlattenVelocities(velocities);
            data.setTemperature(temperature);
            data.setNoseHooverMomentumEnergy(noseHooverMomentumEnergy);
            data.setNoseHooverFrictionEnergy(noseHooverFrictionEnergy);
        }
    }
#else
    void RingPolymerQMMDEngine::applyThermostat()
    {
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &bead = getRingPolymerBeads()[i];
            auto &data = getRingPolymerBeadsPhysicalData()[i];

            getThermostat().applyThermostat(bead, data);
        }
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
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            if (i % mpi::MPI::getSize() == mpi::MPI::getRank())
            {
                auto &bead = getRingPolymerBeads()[i];
                auto &data = getRingPolymerBeadsPhysicalData()[i];

                getManostat().applyManostat(bead, data);
            }
        }

        ::MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto positions = getRingPolymerBeads()[i].getFlattenedQMPositions();
            auto velocities = getRingPolymerBeads()[i].flattenVelocities();

            auto &box = getRingPolymerBeads()[i].getBox();

            const auto oldBoxDimensions = box.get_boxDimensions();
            auto       boxDimensions    = oldBoxDimensions.toStdVector();
            auto volume  = getRingPolymerBeadsPhysicalData()[i].getVolume();
            auto density = getRingPolymerBeadsPhysicalData()[i].getDensity();

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

            const auto boxDimensionsVec = linalg::Vec3D(
                boxDimensions[0],
                boxDimensions[1],
                boxDimensions[2]
            );

            getRingPolymerBeads()[i].deFlattenVelocities(velocities);
            getRingPolymerBeads()[i].deFlattenPositions(positions);

            const auto scalingTensor =
                linalg::diagonalMatrix(boxDimensionsVec / oldBoxDimensions);
            box.scaleBox(scalingTensor);

            getRingPolymerBeadsPhysicalData()[i].setVolume(volume);
            getRingPolymerBeadsPhysicalData()[i].setDensity(density);
        }
    }
#else
    void RingPolymerQMMDEngine::applyManostat()
    {
        for (size_t i = 0; i < getRingPolymerBeads().size(); ++i)
        {
            auto &bead = getRingPolymerBeads()[i];
            auto &data = getRingPolymerBeadsPhysicalData()[i];

            getManostat().applyManostat(bead, data);
        }
    }
#endif

}   // namespace engine
