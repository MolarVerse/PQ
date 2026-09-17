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

#include <cstddef>   // for size_t
#include <cstdint>
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
    enum class ThermostatType : std::uint8_t
    {
        NONE,
        BERENDSEN,
        VELOCITY_RESCALING,
        LANGEVIN,
        NOSE_HOOVER
    };

    [[nodiscard]] std::string string(const ThermostatType& thermostatType);

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
        static inline size_t _nhChainLength            = defaults::NH_CHAIN_LENGTH_DEFAULT;
        static inline size_t _temperatureRampSteps     = 0;
        static inline size_t _temperatureRampFrequency = 1;
        // clang-format on

        static inline double _targetTemperature;
        static inline double _actualTargetTemperature;   // for reset kinetics
        static inline double _startTemperature;
        static inline double _endTemperature;

        // clang-format off
        static inline double _relaxationTime = defaults::BERENDSEN_THERMOSTAT_RELAX_TIME;
        static inline double _friction       = defaults::LANGEVIN_THERMOSTAT_FRICTION;
        static inline double _nhCouplingFreq = defaults::NH_COUPLING_FREQ;
        // clang-format on

        static inline std::map<size_t, double> _chi;
        static inline std::map<size_t, double> _zeta;

       public:
        ThermostatSettings()  = default;
        ~ThermostatSettings() = default;

        static auto addChi(size_t index, double chi)
            -> decltype(_chi.try_emplace(index, chi));
        static auto addZeta(size_t index, double zeta)
            -> decltype(_zeta.try_emplace(index, zeta));

        /***************************
         * standard setter methods *
         ***************************/

        static void setThermostatType(const std::string_view& thermostatType);
        static void setThermostatType(ThermostatType thermostatType);

        static void setTemperatureSet(bool);
        static void setStartTemperatureSet(bool);
        static void setEndTemperatureSet(bool);
        static void setTargetTemperature(double);
        static void setActualTargetTemperature(double);
        static void setStartTemperature(double);
        static void setEndTemperature(double);

        static void setTemperatureRampSteps(size_t);
        static void setTemperatureRampFrequency(size_t);

        static void setRelaxationTime(double);
        static void setFriction(double);
        static void setNoseHooverChainLength(size_t);
        static void setNoseHooverCouplingFrequency(double);

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
