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

#include "constants/conversionFactors.hpp"
#include "constants/internalConversionFactors.hpp"
#include "constants/natureConstants.hpp"
#include "resetKinetics.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"

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
     * @param simulationBox
     */
    void MaxwellBoltzmann::initializeVelocities(
        molsys::SimulationBox &simulationBox
    )
    {
        auto generateVelocities = [this](auto &atom)
        {
            const auto mass              = atom->getMass() * AMU_TO_KG;
            const auto boltzmannConstant = BOLTZMANN_CONSTANT;
            const auto temp =
                settings::ThermostatSettings::getActualTargetTemperature();

            const auto stddev =
                ::sqrt(boltzmannConstant * temp / mass) / VELOCITY_UNIT_TO_SI;

            atom->setVelocity(
                {_randomNumberGenerator.getNormalDistribution(0.0, stddev),
                 _randomNumberGenerator.getNormalDistribution(0.0, stddev),
                 _randomNumberGenerator.getNormalDistribution(0.0, stddev)}
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

        resetKinetics::ResetKinetics::resetMomentum(
            simulationBox,
            simulationBox.calculateMomentum()
        );
        resetKinetics::ResetKinetics::resetAngularMomentum(
            simulationBox,
            simulationBox.calculateAngularMomentum(
                simulationBox.calculateMomentum()
            )
        );
        resetKinetics::ResetKinetics::resetTemperature(
            simulationBox,
            simulationBox.calculateTemperature()
        );
    }

}   // namespace maxwellBoltzmann
