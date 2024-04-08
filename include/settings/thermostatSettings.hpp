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

#ifndef _THERMOSTAT_SETTINGS_HPP_

#define _THERMOSTAT_SETTINGS_HPP_

#include "defaults.hpp"

#include <cstddef>       // for size_t
#include <map>           // for map
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
        LANGEVIN,
        NOSE_HOOVER
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

        static inline size_t _noseHooverChainLength = defaults::_NOSE_HOOVER_CHAIN_LENGTH_DEFAULT_;   // 3

        static inline double _targetTemperature;   // no default value - has to be set by user
        static inline double _relaxationTime              = defaults::_BERENDSEN_THERMOSTAT_RELAXATION_TIME_;   // 0.1 ps
        static inline double _friction                    = defaults::_LANGEVIN_THERMOSTAT_FRICTION_;           // 10.0 ps^-1
        static inline double _noseHooverCouplingFrequency = defaults::_NOSE_HOOVER_COUPLING_FREQUENCY_;         // 1.0e6

        static inline std::map<size_t, double> _chi;    // no default value - has to be set by user
        static inline std::map<size_t, double> _zeta;   // no default value - has to be set by user

      public:
        ThermostatSettings()  = default;
        ~ThermostatSettings() = default;

        static void setThermostatType(const std::string_view &thermostatType);
        static void setThermostatType(const ThermostatType &thermostatType) { _thermostatType = thermostatType; }

        /***************************
         * standard setter methods *
         ***************************/

        static void setNoseHooverChainLength(const size_t length) { _noseHooverChainLength = length; }

        static void setTemperatureSet(const bool temperatureSet) { _isTemperatureSet = temperatureSet; }
        static void setTargetTemperature(const double targetTemperature) { _targetTemperature = targetTemperature; }
        static void setRelaxationTime(const double relaxationTime) { _relaxationTime = relaxationTime; }
        static void setFriction(const double friction) { _friction = friction; }
        static void setNoseHooverCouplingFrequency(const double frequency) { _noseHooverCouplingFrequency = frequency; }

        static auto addChi(const size_t index, const double chi) { return _chi.try_emplace(index, chi); }
        static auto addZeta(const size_t index, const double zeta) { return _zeta.try_emplace(index, zeta); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static size_t getNoseHooverChainLength() { return _noseHooverChainLength; }

        [[nodiscard]] static ThermostatType getThermostatType() { return _thermostatType; }
        [[nodiscard]] static bool           isTemperatureSet() { return _isTemperatureSet; }
        [[nodiscard]] static double         getTargetTemperature() { return _targetTemperature; }
        [[nodiscard]] static double         getRelaxationTime() { return _relaxationTime; }
        [[nodiscard]] static double         getFriction() { return _friction; }
        [[nodiscard]] static double         getNoseHooverCouplingFrequency() { return _noseHooverCouplingFrequency; }

        [[nodiscard]] static std::map<size_t, double> getChi() { return _chi; }
        [[nodiscard]] static std::map<size_t, double> getZeta() { return _zeta; }
    };
}   // namespace settings

#endif   // _THERMOSTAT_SETTINGS_HPP_