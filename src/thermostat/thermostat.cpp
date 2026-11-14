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

#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace thermostat
{

    /**
     * @brief throw if the system has no degrees of freedom
     *
     * @details a system with zero degrees of freedom (for example a single
     * atom with the centre of mass removed) has no temperature to couple to
     *
     * @param simulationBox
     * @param thermostatName name used in the error message
     *
     * @throw exc::UserInputException if there are no degrees of freedom
     */
    void Thermostat::_requireDegreesOfFreedom(
        const molsys::SimulationBox &simulationBox,
        const std::string           &thermostatName
    )
    {
        if (simulationBox.getDegreesOfFreedom() == 0)
        {
            throw exc::UserInputException(
                "Cannot apply " + thermostatName +
                " to a system with zero degrees of freedom."
            );
        }
    }

    /**
     * @brief apply thermostat - base class
     *
     * @note here base class represents none thermostat
     *
     * @param simulationBox
     * @param physicalData
     * @param targetTemperature The target temperature for the system
     */
    void Thermostat::applyThermostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const Temperature & /*targetTemperature*/
    )
    {
        auto _ = scopedTimer(TimerId::Thermostat, "Calc Temperature");

        physicalData.calculateTemperature(simulationBox);
    }

    /**
     * @brief set temperature
     *
     * @param temperature in Kelvin
     */
    void Thermostat::setTemperature(const Temperature &temperature)
    {
        _temperature = temperature;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get temperature
     *
     * @return const Temperature&
     */
    const Temperature &Thermostat::getTemperature() const
    {
        return _temperature;
    }

    /**
     * @brief get the ThermostatType
     *
     * @return ThermostatType
     */
    ThermostatType Thermostat::getThermostatType() const
    {
        return ThermostatType::NONE;
    }

}   // namespace thermostat
