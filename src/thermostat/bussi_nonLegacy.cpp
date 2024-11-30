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

#include <cmath>    // for sqrt
#include <memory>   // for __shared_ptr_access, shared_ptr
#include <vector>   // for vector

#include "atom.hpp"                 // for Atom
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "thermostatSettings.hpp"   // for ThermostatType
#include "timingsSettings.hpp"      // for TimingsSettings
#include "velocityRescalingThermostat.hpp"

using thermostat::VelocityRescalingThermostat;
using namespace settings;
using namespace simulationBox;
using namespace physicalData;

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

    _temperature = simulationBox.calculateTemperature();

    const auto timeStep  = TimingsSettings::getTimeStep();
    const auto tempRatio = _targetTemperature / _temperature;
    const auto dof       = double(simulationBox.getDegreesOfFreedom());

    const auto random = std::normal_distribution<double>(0.0, 1.0)(_generator);

    auto rescalingFactor  = 2.0 * ::sqrt(timeStep * tempRatio / (dof * _tau));
    rescalingFactor      *= random;

    auto lambda  = 1.0 + timeStep / _tau * (tempRatio - 1.0);
    lambda      += rescalingFactor;

    const auto berendsenFactor = ::sqrt(lambda);

    simulationBox.flattenVelocities();
    auto *const velPtr = simulationBox.getVelPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(velPtr)
    for (size_t i = 0; i < simulationBox.getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            velPtr[i*3 + j] *= berendsenFactor;

    // clang-format on

    simulationBox.deFlattenVelocities();

    const auto temperature = _temperature * berendsenFactor * berendsenFactor;

    physicalData.setTemperature(temperature);

    stopTimingsSection("Velocity Rescaling");
}