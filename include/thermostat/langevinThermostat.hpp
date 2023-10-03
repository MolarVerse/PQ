/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "thermostat.hpp"

#include <random>   // for std::random_device, std::mt19937

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

    class LangevinThermostat : public Thermostat
    {
      private:
        std::random_device _randomDevice{};
        std::mt19937       _generator{_randomDevice()};

        double _friction = 0.0;
        double _sigma    = 0.0;

      public:
        LangevinThermostat() = default;
        explicit LangevinThermostat(const double targetTemperature, const double friction);
        LangevinThermostat(const LangevinThermostat &);

        void applyLangevin(simulationBox::SimulationBox &);

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;
        void applyThermostatHalfStep(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] double getFriction() const { return _friction; }
        [[nodiscard]] double getSigma() const { return _sigma; }

        void setFriction(const double friction) { _friction = friction; }
        void setSigma(const double sigma) { _sigma = sigma; }
    };

}   // namespace thermostat

#endif   // _LANGEVIN_THERMOSTAT_HPP_