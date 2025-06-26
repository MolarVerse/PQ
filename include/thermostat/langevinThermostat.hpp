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

#ifndef _LANGEVIN_THERMOSTAT_HPP_

#define _LANGEVIN_THERMOSTAT_HPP_

#include "randomNumberGenerator.hpp"   // for RandomNumberGenerator
#include "thermostat.hpp"
#include "typeAliases.hpp"

using namespace randomNumberGenerator;

namespace thermostat
{

    class LangevinThermostat : public Thermostat
    {
       private:
        RandomNumberGenerator _randomNumberGenerator{};

        double _friction = 0.0;
        double _sigma    = 0.0;

       public:
        explicit LangevinThermostat(const double, const double);
        LangevinThermostat(const LangevinThermostat &);
        LangevinThermostat() = default;

        void calculateSigma(const double, const double);

        void applyLangevin(pq::SimBox &);
        void applyThermostat(pq::SimBox &, pq::PhysicalData &) override;
        void applyThermostatHalfStep(pq::SimBox &, pq::PhysicalData &) override;

        /***************************
         * standard setter methods *
         ***************************/

        void setTargetTemperature(const double targetTemperature) override;

        void setFriction(const double friction);
        void setSigma(const double sigma);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double             getFriction() const;
        [[nodiscard]] double             getSigma() const;
        [[nodiscard]] pq::ThermostatType getThermostatType() const override;
    };

}   // namespace thermostat

#endif   // _LANGEVIN_THERMOSTAT_HPP_