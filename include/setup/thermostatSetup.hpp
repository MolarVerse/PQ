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

#ifndef _THERMOSTAT_SETUP_HPP_

#define _THERMOSTAT_SETUP_HPP_

#include "typeAliases.hpp"

namespace setup
{
    void setupThermostat(pq::Engine &);

    /**
     * @class ThermostatSetup
     *
     * @brief this class setups up the thermostat for the simulation
     *
     */
    class ThermostatSetup
    {
       private:
        pq::MDEngine &_engine;

       public:
        explicit ThermostatSetup(pq::MDEngine &engine);

        void setup();

        void isTargetTemperatureSet() const;
        void setupTemperatureRamp();
        void setupBerendsenThermostat();
        void setupLangevinThermostat();
        void setupNoseHooverThermostat();
        void setupVelocityRescalingThermostat();

        void writeSetupInfo() const;
        void writeBerendsenInfo() const;
        void writeVelocityRescalingInfo() const;
        void writeLangevinInfo() const;
        void writeNoseHooverInfo() const;
        void writeTemperatureRampInfo() const;

        [[nodiscard]] pq::MDEngine &getEngine() const;
    };

}   // namespace setup

#endif   // _THERMOSTAT_SETUP_HPP_