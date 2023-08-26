#ifndef _THERMOSTAT_SETTINGS_HPP_

#define _THERMOSTAT_SETTINGS_HPP_

#include "defaults.hpp"

#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @class ThermostatSettings
     *
     * @brief static class to store settings of the thermostat
     *
     */
    class ThermostatSettings
    {
      private:
        static inline std::string _thermostatType = defaults::_MANOSTAT_DEFAULT_;   // none

        static inline bool _isTemperatureSet = false;

        static inline double _targetTemperature;   // no default value - has to be set by user
        static inline double _relaxationTime = defaults::_BERENDSEN_THERMOSTAT_RELAXATION_TIME_;   // 0.1 ps

      public:
        static void setThermostatType(const std::string_view &thermostatType) { _thermostatType = thermostatType; }
        static void setTemperatureSet(const bool temperatureSet) { _isTemperatureSet = temperatureSet; }
        static void setTargetTemperature(const double targetTemperature) { _targetTemperature = targetTemperature; }
        static void setRelaxationTime(const double relaxationTime) { _relaxationTime = relaxationTime; }

        [[nodiscard]] static std::string getThermostatType() { return _thermostatType; }
        [[nodiscard]] static bool        isTemperatureSet() { return _isTemperatureSet; }
        [[nodiscard]] static double      getTargetTemperature() { return _targetTemperature; }
        [[nodiscard]] static double      getRelaxationTime() { return _relaxationTime; }
    };
}   // namespace settings

#endif   // _THERMOSTAT_SETTINGS_HPP_