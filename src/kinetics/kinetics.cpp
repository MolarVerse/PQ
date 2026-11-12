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
          _temperatureIncrease(settings.getTemperatureIncrease())
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
