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

#ifndef _THERMOSTAT_HPP_

#define _THERMOSTAT_HPP_

#include <cstddef>   // for size_t

#include "timer.hpp"   // for Timer
#include "typeAliases.hpp"

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
    class Thermostat : public timings::Timer
    {
       protected:
        double _temperature       = 0.0;
        double _targetTemperature = 0.0;

        double _temperatureIncrease = 0.0;
        size_t _rampingStepsLeft    = 0;
        size_t _rampingFrequency    = 0;

       public:
        explicit Thermostat(const double targetTemperature);

        Thermostat()          = default;
        virtual ~Thermostat() = default;

        void applyTemperatureRamping();

        virtual void applyThermostat(pq::SimBox &, pq::PhysicalData &);
        virtual void applyThermostatOnForces(pq::SimBox &) {}
        virtual void applyThermostatHalfStep(pq::SimBox &, pq::PhysicalData &)
        {
        }

        /***************************
         * standard setter methods *
         ***************************/

        virtual void setTargetTemperature(const double targetTemperature);
        void         setTemperatureIncrease(const double temperatureIncrease);
        void         setTemperatureRampingSteps(const size_t steps);
        void         setTemperatureRampingFrequency(const size_t frequency);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double getTemperature() const;

        [[nodiscard]] double getTargetTemperature() const;
        [[nodiscard]] double getTemperatureIncrease() const;
        [[nodiscard]] size_t getRampingStepsLeft() const;
        [[nodiscard]] size_t getRampingFrequency() const;

        [[nodiscard]] virtual pq::ThermostatType getThermostatType() const;
    };

}   // namespace thermostat

#endif   // _THERMOSTAT_HPP_
