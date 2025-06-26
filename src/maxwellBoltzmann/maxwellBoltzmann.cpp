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

#include <algorithm>    // for __for_each_fn
#include <cmath>        // for sqrt
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"           // for _AMU_TO_KG_
#include "constants/internalConversionFactors.hpp"   // for _VELOCITY_UNIT_TO_SI_
#include "constants/natureConstants.hpp"             // for _BOLTZMANN_CONSTANT_
#include "resetKinetics.hpp"                         // for ResetKinetics
#include "simulationBox.hpp"                         // for SimulationBox
#include "thermostatSettings.hpp"                    // for ThermostatSettings

#ifdef WITH_MPI
#include <mpi.h>   // for MPI_Bcast, MPI_DOUBLE, MPI_COMM_WORLD

#include "mpi.hpp"   // for MPI
#endif

using maxwellBoltzmann::MaxwellBoltzmann;
using namespace simulationBox;
using namespace constants;
using namespace settings;
using namespace resetKinetics;

/**
 * @brief generate boltzmann distributed velocities for all atoms in the
 * simulation box
 *
 * @details using a standard deviation of sqrt(kb*T/m) for each component of the
 * velocity vector
 *
 * @param simBox
 */
void MaxwellBoltzmann::initializeVelocities(SimulationBox &simBox)
{
    auto generateVelocities = [this](auto &atom)
    {
        const auto mass = atom->getMass() * _AMU_TO_KG_;
        const auto kb   = _BOLTZMANN_CONSTANT_;
        const auto temp = ThermostatSettings::getActualTargetTemperature();

        const auto stddev = ::sqrt(kb * temp / mass) / _VELOCITY_UNIT_TO_SI_;

        atom->setVelocity(
            {_randomNumberGenerator.getNormalDistribution(0.0, stddev),
             _randomNumberGenerator.getNormalDistribution(0.0, stddev),
             _randomNumberGenerator.getNormalDistribution(0.0, stddev)}
        );
    };

#ifdef WITH_MPI
    if (mpi::MPI::isRoot())
        std::ranges::for_each(simBox.getAtoms(), generateVelocities);

    auto velocities = simBox.flattenVelocities();

    ::MPI_Bcast(
        velocities.data(),
        velocities.size(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    simBox.deFlattenVelocities(velocities);
#else
    std::ranges::for_each(simBox.getAtoms(), generateVelocities);
#endif

    auto resetKinetics = ResetKinetics();
    resetKinetics.setMomentum(simBox.calculateMomentum());
    resetKinetics.resetMomentum(simBox);
    resetKinetics.resetAngularMomentum(simBox);
    resetKinetics.resetTemperature(simBox);
}