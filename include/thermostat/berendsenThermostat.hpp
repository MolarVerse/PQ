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

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

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
        BerendsenThermostat() = default;
        explicit BerendsenThermostat(const double targetTemp, const double tau)
            : Thermostat(targetTemp), _tau(tau)
        {
        }

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &)
            override;

        [[nodiscard]] pq::ThermostatType getThermostatType() const override;
        [[nodiscard]] double             getTau() const { return _tau; }
        void setTau(const double tau) { _tau = tau; }
    };

}   // namespace thermostat

#endif   // _BERENDSEN_THERMOSTAT_HPP_