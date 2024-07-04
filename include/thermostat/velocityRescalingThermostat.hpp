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

#ifndef _VELOCITY_RESCALING_THERMOSTAT_HPP_

#define _VELOCITY_RESCALING_THERMOSTAT_HPP_

#include <random>   // for std::random_device, std::mt19937

#include "thermostat.hpp"
#include "typeAliases.hpp"

namespace thermostat
{
    /**
     * @class VelocityRescalingThermostat
     *
     * @brief this class implements the velocity rescaling thermostat
     * and inherits from the Thermostat class
     *
     */
    class VelocityRescalingThermostat : public Thermostat
    {
       private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _tau = 0.0;

       public:
        VelocityRescalingThermostat() = default;
        VelocityRescalingThermostat(const VelocityRescalingThermostat &);
        explicit VelocityRescalingThermostat(const double, const double);

        void applyThermostat(pq::SimBox &, pq::PhysicalData &) override;

        [[nodiscard]] double getTau() const;
        void                 setTau(const double tau);
    };
}   // namespace thermostat

#endif   // _VELOCITY_RESCALING_THERMOSTAT_HPP_