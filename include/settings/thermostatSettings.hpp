#ifndef _THERMOSTAT_SETTINGS_HPP_

#define _THERMOSTAT_SETTINGS_HPP_

#include "defaults.hpp"

#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{

    /**
     * @enum ThermostatType
     *
     * @brief enum class to store the type of thermostat
     *
     */
    enum class ThermostatType
    {
        NONE,
        BERENDSEN,
        VELOCITY_RESCALING,
        LANGEVIN
    };

    [[nodiscard]] std::string string(const ThermostatType &thermostatType);

    /**
     * @class ThermostatSettings
     *
     * @brief static class to store settings of the thermostat
     *
     */
    class ThermostatSettings
    {
      private:
        static inline ThermostatType _thermostatType = ThermostatType::NONE;

        static inline bool _isTemperatureSet = false;

        static inline double _targetTemperature;   // no default value - has to be set by user
        static inline double _relaxationTime = defaults::_BERENDSEN_THERMOSTAT_RELAXATION_TIME_;   // 0.1 ps
        static inline double _friction       = defaults::_LANGEVIN_THERMOSTAT_FRICTION_;           // 10.0 ps^-1

      public:
        ThermostatSettings()  = default;
        ~ThermostatSettings() = default;

        static void setThermostatType(const std::string_view &thermostatType);

        static void setThermostatType(const ThermostatType &thermostatType) { _thermostatType = thermostatType; }
        static void setTemperatureSet(const bool temperatureSet) { _isTemperatureSet = temperatureSet; }
        static void setTargetTemperature(const double targetTemperature) { _targetTemperature = targetTemperature; }
        static void setRelaxationTime(const double relaxationTime) { _relaxationTime = relaxationTime; }
        static void setFriction(const double friction) { _friction = friction; }

        [[nodiscard]] static ThermostatType getThermostatType() { return _thermostatType; }
        [[nodiscard]] static bool           isTemperatureSet() { return _isTemperatureSet; }
        [[nodiscard]] static double         getTargetTemperature() { return _targetTemperature; }
        [[nodiscard]] static double         getRelaxationTime() { return _relaxationTime; }
        [[nodiscard]] static double         getFriction() { return _friction; }
    };
}   // namespace settings

#endif   // _THERMOSTAT_SETTINGS_HPP_