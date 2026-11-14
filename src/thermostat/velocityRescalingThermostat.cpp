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

#include "velocityRescalingThermostat.hpp"

#include <cmath>

#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "mathUtilities.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace thermostat
{

    /**
     * @brief Construct a new Velocity Rescaling Thermostat:: Velocity Rescaling
     * Thermostat object
     *
     * @param tau
     */
    VelocityRescalingThermostat::VelocityRescalingThermostat(const Time &tau)
        : _tau(tau)
    {
    }

    /**
     * @brief Copy constructor for Velocity Rescaling Thermostat
     *
     * @param other
     */
    VelocityRescalingThermostat::VelocityRescalingThermostat(
        const VelocityRescalingThermostat &other
    )
        : Thermostat(other), _tau(other._tau)
    {
    }

    /**
     * @brief Copy assignment operator for Velocity Rescaling Thermostat
     *
     * @param other
     * @return VelocityRescalingThermostat&
     */
    VelocityRescalingThermostat &VelocityRescalingThermostat::operator=(
        const VelocityRescalingThermostat &other
    )
    {
        if (this != &other)
        {
            Thermostat::operator=(other);
            _tau = other._tau;
        }
        return *this;
    }

    /**
     * @brief apply thermostat - [Velocity
     * Rescaling](https://doi.org/10.1063/1.2408420)
     *
     * @param simulationBox
     * @param physicalData
     * @param targetTemperature The target temperature for the system
     */
    void VelocityRescalingThermostat::applyThermostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const Temperature          &targetTemperature
    )
    {
        auto _ = scopedTimer(TimerId::Thermostat, "Velocity Rescaling");

        _requireDegreesOfFreedom(simulationBox, "velocity rescaling");

        physicalData.calculateTemperature(simulationBox);

        const auto temp = physicalData.getTemperature();
        setTemperature(temp);

        if (utilities::isZero(getTemperature().quantity_from_zero()))
        {
            if (utilities::isZero(targetTemperature.quantity_from_zero()))
                return;

            throw exc::UserInputException(
                "Cannot apply velocity rescaling to a zero-temperature system "
                "with a positive target temperature. Initialize velocities "
                "first."
            );
        }

        const auto timeStep  = settings::TimingsSettings::getTimeStep();
        const auto timeRatio = static_cast<double>(timeStep / _tau);
        const auto tempRatio = targetTemperature.quantity_from_zero() /
                               getTemperature().quantity_from_zero();
        const auto dof =
            static_cast<double>(simulationBox.getDegreesOfFreedom());

        auto lambda =
            1.0 + (timeRatio * (tempRatio - 1.0)).numerical_value_in(one);

        const auto rescalingFactor =
            2.0 * ::sqrt(timeRatio * tempRatio.numerical_value_in(one) / dof);

        auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        lambda += rescalingFactor * random;

        // If the stochastic term makes lambda negative, draw a new random
        // number until lambda is positive to prevent sqrt of a negative number,
        // leading to -nan velocities
        while (lambda < 0.0)
        {
            lambda -= rescalingFactor * random;
            random  = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);
            lambda += rescalingFactor * random;
        }

        const auto berendsenFactor = ::sqrt(lambda);

        for (const auto &atom : simulationBox.getAtoms())
            atom->scaleVelocity(berendsenFactor);

        const Temperature temperature{
            getTemperature().quantity_from_zero() * berendsenFactor *
            berendsenFactor
        };

        physicalData.setTemperature(temperature);
    }

    /**
     * @brief Get the tau (relaxation time) of the Velocity Rescaling thermostat
     *
     * @return const Time&
     */
    const Time &VelocityRescalingThermostat::getTau() const { return _tau; }

    /**
     * @brief Set the tau (relaxation time) of the Velocity Rescaling thermostat
     *
     * @param tau
     */
    void VelocityRescalingThermostat::setTau(const Time &tau) { _tau = tau; }
    /**
     * @brief Get thermostat type
     *
     * @return ThermostatType
     */
    ThermostatType VelocityRescalingThermostat::getThermostatType() const
    {
        return ThermostatType::VELOCITY_RESCALING;
    }

}   // namespace thermostat
