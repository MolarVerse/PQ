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

#include <cmath>    // for sqrt
#include <memory>   // for __shared_ptr_access, shared_ptr
#include <vector>   // for vector

#include "atom.hpp"                 // for Atom
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "thermostatSettings.hpp"   // for ThermostatType
#include "timingsSettings.hpp"      // for TimingsSettings

using thermostat::VelocityRescalingThermostat;
using namespace settings;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Construct a new Velocity Rescaling Thermostat:: Velocity Rescaling
 * Thermostat object
 *
 * @param targetTemp
 * @param tau
 */
VelocityRescalingThermostat::VelocityRescalingThermostat(
    const double targetTemp,
    const double tau
)
    : Thermostat(targetTemp), _tau(tau)
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
 * @brief apply thermostat - Velocity Rescaling
 *
 * @link https://doi.org/10.1063/1.2408420
 *
 * @param simulationBox
 * @param physicalData
 */
void VelocityRescalingThermostat::applyThermostat(
    SimulationBox &simulationBox,
    PhysicalData  &physicalData
)
{
    startTimingsSection("Velocity Rescaling");

    physicalData.calculateTemperature(simulationBox);

    _temperature = physicalData.getTemperature();

    const auto timeStep  = TimingsSettings::getTimeStep();
    const auto tempRatio = _targetTemperature / _temperature;
    const auto dof       = double(simulationBox.getDegreesOfFreedom());

    const auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

    auto rescalingFactor  = 2.0 * ::sqrt(timeStep * tempRatio / (dof * _tau));
    rescalingFactor      *= random;

    auto lambda  = 1.0 + timeStep / _tau * (tempRatio - 1.0);
    lambda      += rescalingFactor;

    const auto berendsenFactor = ::sqrt(lambda);

    for (const auto &atom : simulationBox.getAtoms())
        atom->scaleVelocity(berendsenFactor);

    const auto temperature = _temperature * berendsenFactor * berendsenFactor;

    physicalData.setTemperature(temperature);

    stopTimingsSection("Velocity Rescaling");
}

/**
 * @brief Get the tau (relaxation time) of the Velocity Rescaling thermostat
 *
 * @return double
 */
double VelocityRescalingThermostat::getTau() const { return _tau; }

/**
 * @brief Set the tau (relaxation time) of the Velocity Rescaling thermostat
 *
 * @param tau
 */
void VelocityRescalingThermostat::setTau(const double tau) { _tau = tau; }
/**
 * @brief Get thermostat type
 *
 * @return ThermostatType
 */
ThermostatType VelocityRescalingThermostat::getThermostatType() const
{
    return ThermostatType::VELOCITY_RESCALING;
}