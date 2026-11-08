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

#include "berendsenThermostat.hpp"

#include <cmath>

#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "mathUtilities.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace thermostat
{

    /**
     * @brief Construct a new Berendsen Thermostat object
     *
     * @param targetTemp
     * @param tau
     */
    BerendsenThermostat::BerendsenThermostat(
        const Temperature &targetTemp,
        const Time        &tau
    )
        : Thermostat(targetTemp), _tau(tau)
    {
    }

    /**
     * @brief apply thermostat - [Berendsen](https://doi.org/10.1063/1.448118)
     *
     * @param simulationBox
     * @param physicalData
     */
    void BerendsenThermostat::applyThermostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        auto _ = scopedTimer(TimerId::Thermostat, "Berendsen");

        physicalData.calculateTemperature(simulationBox);

        setTemperature(physicalData.getTemperature());

        if (utilities::isZero(getTemperature().quantity_from_zero()))
        {
            if (utilities::isZero(getTargetTemperature().quantity_from_zero()))
                return;

            throw exc::UserInputException(
                "Cannot apply Berendsen coupling to a zero-temperature system "
                "with a positive target temperature. Initialize velocities "
                "first."
            );
        }

        const auto &timeStep  = settings::TimingsSettings::getTimeStep();
        const auto  timeRatio = static_cast<double>(timeStep / _tau);
        const auto  tempRatio = getTargetTemperature().quantity_from_zero() /
                               getTemperature().quantity_from_zero();

        const auto berendsenFactor = ::sqrt(
            1.0 + (timeRatio * (tempRatio - 1.0)).numerical_value_in(one)
        );

        for (const auto &atom : simulationBox.getAtoms())
            atom->scaleVelocity(berendsenFactor);

        physicalData.setTemperature(
            Temperature{
                getTemperature().quantity_from_zero() * berendsenFactor *
                berendsenFactor
            }
        );
    }

    /**
     * @brief Get the tau (relaxation time) of the Berendsen thermostat
     *
     * @return const Time &
     */
    const Time &BerendsenThermostat::getTau() const { return _tau; }

    /**
     * @brief Set the tau (relaxation time) of the Berendsen thermostat
     *
     * @param tau
     */
    void BerendsenThermostat::setTau(const Time &tau) { _tau = tau; }

    /**
     * @brief Get thermostat type
     *
     * @return ThermostatType
     */
    ThermostatType BerendsenThermostat::getThermostatType() const
    {
        return ThermostatType::BERENDSEN;
    }

}   // namespace thermostat
