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

#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "simulationBox_API.hpp"    // for calculateTemperature
#include "thermostatSettings.hpp"   // for ThermostatType
#include "timingsSettings.hpp"      // for TimingsSettings

using thermostat::BerendsenThermostat;
using namespace simulationBox;
using namespace physicalData;
using namespace settings;

/**
 * @brief Construct a new Berendsen Thermostat object
 *
 * @param targetTemp
 * @param tau
 */
BerendsenThermostat::BerendsenThermostat(
    const double targetTemp,
    const double tau
)
    : Thermostat(targetTemp), _tau(tau)
{
}

/**
 * @brief apply thermostat - Berendsen
 *
 * @link https://doi.org/10.1063/1.448118
 *
 * @param simulationBox
 * @param data
 */
void BerendsenThermostat::applyThermostat(
    SimulationBox &simulationBox,
    PhysicalData  &data
)
{
    startTimingsSection("Berendsen");

    _temperature = calculateTemperature(simulationBox);

    const auto dt        = TimingsSettings::getTimeStep();
    const auto tempRatio = _targetTemperature / _temperature;

    const auto berendsenFactor = ::sqrt(1.0 + dt / _tau * (tempRatio - 1.0));

    simulationBox.scaleVelocities(berendsenFactor);

    data.setTemperature(_temperature * berendsenFactor * berendsenFactor);

    stopTimingsSection("Berendsen");
}

/**
 * @brief Get the tau (relaxation time) of the Berendsen thermostat
 *
 * @return double
 */
double BerendsenThermostat::getTau() const { return _tau; }

/**
 * @brief Set the tau (relaxation time) of the Berendsen thermostat
 *
 * @param tau
 */
void BerendsenThermostat::setTau(const double tau) { _tau = tau; }

/**
 * @brief Get thermostat type
 *
 * @return ThermostatType
 */
ThermostatType BerendsenThermostat::getThermostatType() const
{
    return ThermostatType::BERENDSEN;
}