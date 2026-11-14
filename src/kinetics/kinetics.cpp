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

#include "kinetics.hpp"

#include "thermostatSettings.hpp"

namespace kinetics
{
    /**
     * @brief Constructs a Kinetics object using the provided KineticsSettings
     *
     * @param settings The settings specific to the kinetics
     */
    Kinetics::Kinetics(const KineticsSettings& settings)
        : _settings(settings),
          _currentTargetTemperature(settings.getStartTemperature()),
          _temperatureIncrease(settings.getTemperatureIncrease()),
          _rampingStepsLeft(settings.getRampingSteps())
    {
    }

    /**
     * @brief Apply temperature ramping
     *
     */
    void Kinetics::applyTemperatureRamping()
    {
        const auto stepsLeft = _rampingStepsLeft;

        if (stepsLeft > 0 &&
            (stepsLeft - 1) % _settings->getRampingFrequency() == 0)
        {
            _currentTargetTemperature += _temperatureIncrease;
        }

        if (_rampingStepsLeft > 0)
            --_rampingStepsLeft;
    }

    /**
     * @brief Returns the current target temperature
     *
     * @return The current target temperature
     */
    const Temperature& Kinetics::getCurrentTargetTemperature() const
    {
        return _currentTargetTemperature;
    }
}   // namespace kinetics
