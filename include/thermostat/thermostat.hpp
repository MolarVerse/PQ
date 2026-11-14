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

#ifndef THERMOSTAT__THERMOSTAT_HPP
#define THERMOSTAT__THERMOSTAT_HPP

#include <string>

#include "enums/thermostat.hpp"
#include "unit_types.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace thermostat
{
    /**
     * @class Thermostat
     *
     * @brief Thermostat is a base class for all thermostats
     *
     * @details it provides a dummy function applyThermostat() which does only
     * calculate the temperature
     *
     */
    class Thermostat
    {
       private:
        Temperature _temperature;

       public:
        Thermostat()          = default;
        virtual ~Thermostat() = default;

        virtual void applyThermostat(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            const Temperature &targetTemperature
        );

        virtual void applyThermostatOnForces(
            molsys::SimulationBox & /*simBox*/,
            const Temperature & /*targetTemperature*/
        )
        {
        }

        virtual void applyThermostatHalfStep(
            molsys::SimulationBox & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const Temperature & /*targetTemperature*/
        )
        {
        }

        /***************************
         * standard setter methods *
         ***************************/

        void setTemperature(const Temperature &temperature);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] const Temperature &getTemperature() const;

        [[nodiscard]]
        virtual ThermostatType getThermostatType() const;

       protected:
        static void _requireDegreesOfFreedom(
            const molsys::SimulationBox &simulationBox,
            const std::string           &thermostatName
        );
    };

}   // namespace thermostat

#endif   // THERMOSTAT__THERMOSTAT_HPP
