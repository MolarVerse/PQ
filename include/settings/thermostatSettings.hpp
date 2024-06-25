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

#include <cstddef>       // for size_t
#include <map>           // for map
#include <string>        // for string
#include <string_view>   // for string_view

#include "defaults.hpp"

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

        static inline bool _isTemperatureSet      = false;
        static inline bool _isStartTemperatureSet = false;
        static inline bool _isEndTemperatureSet   = false;

        // clang-format off
        static inline size_t _nhChainLength            = defaults::_NH_CHAIN_LENGTH_DEFAULT_;
        static inline size_t _temperatureRampSteps     = 0;
        static inline size_t _temperatureRampFrequency = 1;
        // clang-format on

        static inline double _targetTemperature;
        static inline double _actualTargetTemperature;   // for reset kinetics
        static inline double _startTemperature;
        static inline double _endTemperature;

        // clang-format off
        static inline double _relaxationTime = defaults::_BERENDSEN_THERMOSTAT_RELAX_TIME_;
        static inline double _friction       = defaults::_LANGEVIN_THERMOSTAT_FRICTION_;
        static inline double _nhCouplingFreq = defaults::_NH_COUPLING_FREQ_;
        // clang-format on

        static inline std::map<size_t, double> _chi;
        static inline std::map<size_t, double> _zeta;

       public:
        ThermostatSettings()  = default;
        ~ThermostatSettings() = default;

        static auto addChi(const size_t index, const double chi);
        static auto addZeta(const size_t index, const double zeta);

        /***************************
         * standard setter methods *
         ***************************/

        static void setThermostatType(const std::string_view &thermostatType);
        static void setThermostatType(const ThermostatType &thermostatType);

        static void setTemperatureSet(const bool);
        static void setStartTemperatureSet(const bool);
        static void setEndTemperatureSet(const bool);
        static void setTargetTemperature(const double);
        static void setActualTargetTemperature(const double);
        static void setStartTemperature(const double);
        static void setEndTemperature(const double);

        static void setTemperatureRampSteps(const size_t);
        static void setTemperatureRampFrequency(const size_t);

        static void setRelaxationTime(const double);
        static void setFriction(const double);
        static void setNoseHooverChainLength(const size_t);
        static void setNoseHooverCouplingFrequency(const double);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static ThermostatType getThermostatType();

        [[nodiscard]] static size_t getNoseHooverChainLength();
        [[nodiscard]] static size_t getTemperatureRampSteps();
        [[nodiscard]] static size_t getTemperatureRampFrequency();

        [[nodiscard]] static bool isTemperatureSet();
        [[nodiscard]] static bool isStartTemperatureSet();
        [[nodiscard]] static bool isEndTemperatureSet();

        [[nodiscard]] static double getTargetTemperature();
        [[nodiscard]] static double getActualTargetTemperature();
        [[nodiscard]] static double getStartTemperature();
        [[nodiscard]] static double getEndTemperature();
        [[nodiscard]] static double getRelaxationTime();
        [[nodiscard]] static double getFriction();
        [[nodiscard]] static double getNoseHooverCouplingFrequency();

        [[nodiscard]] static std::map<size_t, double> getChi();
        [[nodiscard]] static std::map<size_t, double> getZeta();
    };
}   // namespace settings

#endif   // _THERMOSTAT_SETTINGS_HPP_