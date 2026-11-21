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

#ifndef THERMOSTAT__LANGEVIN_THERMOSTAT_HPP
#define THERMOSTAT__LANGEVIN_THERMOSTAT_HPP

#include <mstd/utils.hpp>

#include "randomNumberGenerator.hpp"
#include "thermostat.hpp"
#include "unit_types.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace thermostat
{

    class LangevinThermostat : public Thermostat
    {
       private:
        rng::RandomNumberGenerator _randomNumberGenerator;

        Frequency _friction = Frequency::zero();

       public:
        explicit LangevinThermostat(const Frequency &friction);
        ~LangevinThermostat() override = default;

        // copy constructor and copy assignment needed for random number
        // generator
        LangevinThermostat(const LangevinThermostat &);
        LangevinThermostat &operator=(const LangevinThermostat &);
        MSTD_NO_MOVE(LangevinThermostat);

        [[nodiscard]]
        static auto calculateSigma(
            const Frequency &friction,
            const Temperature &,
            const Time &targetTemperature
        ) -> decltype(mstd::sqrt(EnergyPerMol{} / TimeSquared{}));

        void applyLangevin(
            molsys::SimulationBox &,
            const Temperature &,
            const Time &timeStep
        );

        void applyThermostat(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData,
            const Temperature          &targetTemperature,
            const Time                 &timeStep
        ) override;

        void applyThermostatHalfStep(
            molsys::SimulationBox &simulationBox,
            physicalData::PhysicalData & /*physicalData*/,
            const Temperature &targetTemperature,
            const Time        &timeStep
        ) override;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]]
        ThermostatType getThermostatType() const override;
    };

}   // namespace thermostat

#endif   // THERMOSTAT__LANGEVIN_THERMOSTAT_HPP
