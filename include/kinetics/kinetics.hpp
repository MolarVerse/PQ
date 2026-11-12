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

#ifndef KINETICS__KINETICS_HPP
#define KINETICS__KINETICS_HPP

#include <mstd/types.hpp>

#include "unit_types.hpp"

class KineticsSettings;

namespace kinetics
{
    /**
     * @class Kinetics
     *
     * @brief Represents the kinetics of the system, managing temperature
     * ramping and target temperatures
     *
     */
    class Kinetics
    {
       private:
        mstd::ConstRef<KineticsSettings> _settings;
        Temperature                      _currentTargetTemperature;
        TemperatureDiff                  _temperatureIncrease;
        size_t                           _rampingStepsLeft = 0;

       public:
        explicit Kinetics(const KineticsSettings& settings);

        void applyTemperatureRamping();

        [[nodiscard]]
        const Temperature& getCurrentTargetTemperature() const;
    };
}   // namespace kinetics

#endif   // KINETICS__KINETICS_HPP
