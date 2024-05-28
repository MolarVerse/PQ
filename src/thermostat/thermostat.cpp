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

#include "thermostat.hpp"

#include "physicalData.hpp"         // for physicalData::PhysicalData
#include "thermostatSettings.hpp"   // for settings::ThermostatSettings

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

using thermostat::Thermostat;

/**
 * @brief apply thermostat - base class
 *
 * @note here base class represents none thermostat
 *
 * @param simulationBox
 * @param physicalData
 */
void Thermostat::applyThermostat(
    simulationBox::SimulationBox &simulationBox,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("Calc Temperature");
    physicalData.calculateTemperature(simulationBox);
    stopTimingsSection("Calc Temperature");
}

/**
 * @brief Apply temperature ramping
 *
 */
void Thermostat::applyTemperatureRamping()
{
    if (_rampingStepsLeft > 0 &&
        (_rampingStepsLeft - 1) % _rampingFrequency == 0)
    {
        setTargetTemperature(_targetTemperature + _temperatureIncrease);
        settings::ThermostatSettings::setActualTargetTemperature(
            _targetTemperature
        );
    }

    if (_rampingStepsLeft > 0)
    {
        --_rampingStepsLeft;
    }
}
