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

#ifndef _BERENDSEN_THERMOSTAT_HPP_

#define _BERENDSEN_THERMOSTAT_HPP_

#include "thermostat.hpp"
#include "typeAliases.hpp"

namespace thermostat
{
    /**
     * @class BerendsenThermostat
     *
     * @brief BerendsenThermostat is a class for the Berendsen thermostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class BerendsenThermostat : public Thermostat
    {
       private:
        double _tau;

       public:
        explicit BerendsenThermostat(const double targetTemp, const double tau);
        BerendsenThermostat() = default;

        void applyThermostat(pq::SimBox &, pq::PhysicalData &) override;

        void setTau(const double tau);

        [[nodiscard]] double             getTau() const;
        [[nodiscard]] pq::ThermostatType getThermostatType() const override;
    };

}   // namespace thermostat

#endif   // _BERENDSEN_THERMOSTAT_HPP_