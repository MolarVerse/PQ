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

#include "maxwellBoltzmann.hpp"

#include <algorithm>
#include <cmath>

#include "resetKinetics.hpp"
#include "simulationBox.hpp"
#include "units.hpp"

#ifdef WITH_MPI
#include <mpi.h>

#include "mpi.hpp"
#endif

namespace maxwellBoltzmann
{
    /**
     * @brief generate boltzmann distributed velocities for all atoms in the
     * simulation box
     *
     * @details using a standard deviation of sqrt(kb*T/m) for each component of
     * the velocity vector
     *
     * @param simulationBox The simulation box containing all atoms for which
     * velocities should be initialized
     * @param targetTemperature The target temperature for the system
     */
    void MaxwellBoltzmann::initializeVelocities(
        molsys::SimulationBox &simulationBox,
        const Temperature     &targetTemperature
    )
    {
        auto generateVelocities = [this, &targetTemperature](auto &atom)
        {
            const auto mass = atom->getMass();
            const auto temp = targetTemperature.quantity_from_zero();

            const Vel stddev{mstd::sqrt((temp / mass) * kb)};

            atom->setVelocity(
                Vel3D{
                    _randomNumberGenerator
                        .getNormalDistribution(Vel::zero(), stddev),
                    _randomNumberGenerator
                        .getNormalDistribution(Vel::zero(), stddev),
                    _randomNumberGenerator
                        .getNormalDistribution(Vel::zero(), stddev)
                }
            );
        };

#ifdef WITH_MPI
        if (mpi::MPI::isRoot())
            std::ranges::for_each(simulationBox.getAtoms(), generateVelocities);

        auto velocities = simulationBox.flattenVelocities();

        ::MPI_Bcast(
            velocities.data(),
            velocities.size(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        simulationBox.deFlattenVelocities(velocities);
#else
        std::ranges::for_each(simulationBox.getAtoms(), generateVelocities);
#endif

        kinetics::ResetKinetics::resetMomentum(
            simulationBox,
            simulationBox.calculateMomentum()
        );
        kinetics::ResetKinetics::resetAngularMomentum(
            simulationBox,
            simulationBox.calculateAngularMomentum(
                simulationBox.calculateMomentum()
            )
        );
        kinetics::ResetKinetics::resetTemperature(
            simulationBox,
            simulationBox.calculateTemperature(),
            targetTemperature
        );
    }

}   // namespace maxwellBoltzmann
