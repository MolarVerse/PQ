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

#ifndef THERMOSTAT__NOSE_HOOVER_THERMOSTAT_HPP
#define THERMOSTAT__NOSE_HOOVER_THERMOSTAT_HPP

#include <mstd/types.hpp>
#include <vector>

#include "thermostat.hpp"

class NoseHooverSettings;

namespace thermostat
{
    /**
     * @class NoseHooverThermostat
     *
     * @brief this class implements the Nose-Hoover thermostat
     *
     */
    class NoseHooverThermostat : public Thermostat
    {
       private:
        mstd::ConstRef<NoseHooverSettings> _nhSettings;
        // chainLength + 1 entries; the last one is a fixed zero boundary term
        // needed by the chi[i] * chi[i + 1] coupling of the last real link. It
        // is never updated and never exposed through the getters.
        std::vector<double> _chi{0.0};    // in kcal/mol s
        std::vector<double> _zeta{0.0};   // unitless

       public:
        explicit NoseHooverThermostat(const NoseHooverSettings &nhSettings);

        void applyThermostat(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData,
            const Temperature          &targetTemperature
        ) override;

        void applyThermostatOnForces(
            molsys::SimulationBox &simulationBox,
            const Temperature     &targetTemperature
        ) override;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::vector<double> getChi() const;
        [[nodiscard]] std::vector<double> getZeta() const;

        [[nodiscard]]
        ThermostatType getThermostatType() const override;

       private:
        static void _requireCoupling(
            const molsys::SimulationBox &simulationBox,
            const Temperature           &targetTemperature
        );
    };
}   // namespace thermostat

#endif   // THERMOSTAT__NOSE_HOOVER_THERMOSTAT_HPP
