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

#include "physicalData.hpp"         // for PhysicalData
#include "thermostatSettings.hpp"   // for ThermostatSettings

using thermostat::Thermostat;
using namespace simulationBox;
using namespace physicalData;
using namespace settings;

/**
 * @brief Construct a new Thermostat:: Thermostat object
 *
 * @param targetTemperature
 */
Thermostat::Thermostat(const double targetTemperature)
    : _targetTemperature(targetTemperature)
{
}

/**
 * @brief apply thermostat - base class
 *
 * @note here base class represents none thermostat
 *
 * @param simulationBox
 * @param physicalData
 */
void Thermostat::applyThermostat(
    SimulationBox &simulationBox,
    PhysicalData  &physicalData
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
    const auto stepsLeft = _rampingStepsLeft;

    if (stepsLeft > 0 && (stepsLeft - 1) % _rampingFrequency == 0)
    {
        setTargetTemperature(_targetTemperature + _temperatureIncrease);
        ThermostatSettings::setActualTargetTemperature(_targetTemperature);
    }

    if (_rampingStepsLeft > 0)
        --_rampingStepsLeft;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set target temperature
 *
 * @param targetTemperature
 */
void Thermostat::setTargetTemperature(const double targetTemperature)
{
    _targetTemperature = targetTemperature;
}

/**
 * @brief set temperature increase
 *
 * @param temperatureIncrease
 */
void Thermostat::setTemperatureIncrease(const double temperatureIncrease)
{
    _temperatureIncrease = temperatureIncrease;
}

/**
 * @brief set temperature ramping steps
 *
 * @param steps
 */
void Thermostat::setTemperatureRampingSteps(const size_t steps)
{
    _rampingStepsLeft = steps;
}

/**
 * @brief set temperature ramping frequency
 *
 * @param frequency
 */
void Thermostat::setTemperatureRampingFrequency(const size_t frequency)
{
    _rampingFrequency = frequency;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get temperature
 *
 * @return double
 */
double Thermostat::getTemperature() const { return _temperature; }

/**
 * @brief get target temperature
 *
 * @return double
 */
double Thermostat::getTargetTemperature() const { return _targetTemperature; }

/**
 * @brief get temperature increase
 *
 * @return double
 */
double Thermostat::getTemperatureIncrease() const
{
    return _temperatureIncrease;
}

/**
 * @brief get ramping steps left
 *
 * @return size_t
 */
size_t Thermostat::getRampingStepsLeft() const { return _rampingStepsLeft; }

/**
 * @brief get ramping frequency
 *
 * @return size_t
 */
size_t Thermostat::getRampingFrequency() const { return _rampingFrequency; }
/**
 * @brief get the ThermostatType
 *
 * @return ThermostatType
 */
ThermostatType Thermostat::getThermostatType() const
{
    return ThermostatType::NONE;
}