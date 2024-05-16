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

        static inline size_t _noseHooverChainLength =
            defaults::_NOSE_HOOVER_CHAIN_LENGTH_DEFAULT_;   // 3
        static inline size_t _temperatureRampSteps     = 0;
        static inline size_t _temperatureRampFrequency = 1;

        static inline double _targetTemperature;
        static inline double _actualTargetTemperature;   // for reset kinetics
        static inline double _startTemperature;
        static inline double _endTemperature;

        static inline double _relaxationTime =
            defaults::_BERENDSEN_THERMOSTAT_RELAXATION_TIME_;   // 0.1 ps
        static inline double _friction =
            defaults::_LANGEVIN_THERMOSTAT_FRICTION_;   // 10.0 ps^-1
        static inline double _noseHooverCouplingFrequency =
            defaults::_NOSE_HOOVER_COUPLING_FREQUENCY_;   // 1.0e6

        static inline std::map<size_t, double> _chi;
        static inline std::map<size_t, double> _zeta;

       public:
        ThermostatSettings()  = default;
        ~ThermostatSettings() = default;

        static void setThermostatType(const std::string_view &thermostatType);

        /***************************
         * standard setter methods *
         ***************************/

        // ThermostatType setters

        static void setThermostatType(const ThermostatType &thermostatType)
        {
            _thermostatType = thermostatType;
        }

        // size_t setters

        static void setNoseHooverChainLength(const size_t length)
        {
            _noseHooverChainLength = length;
        }

        static void setTemperatureRampSteps(const size_t steps)
        {
            _temperatureRampSteps = steps;
        }

        static void setTemperatureRampFrequency(const size_t frequency)
        {
            _temperatureRampFrequency = frequency;
        }

        // bool setters

        static void setTemperatureSet(const bool temperatureSet)
        {
            _isTemperatureSet = temperatureSet;
        }

        static void setStartTemperatureSet(const bool startTemperatureSet)
        {
            _isStartTemperatureSet = startTemperatureSet;
        }

        static void setEndTemperatureSet(const bool endTemperatureSet)
        {
            _isEndTemperatureSet = endTemperatureSet;
        }

        // double setters

        static void setTargetTemperature(const double targetTemperature)
        {
            _targetTemperature = targetTemperature;
            setTemperatureSet(true);
            setActualTargetTemperature(targetTemperature);
        }

        static void setActualTargetTemperature(
            const double actualTargetTemperature
        )
        {
            _actualTargetTemperature = actualTargetTemperature;
        }

        static void setStartTemperature(const double startTemperature)
        {
            _startTemperature = startTemperature;
            setStartTemperatureSet(true);
        }

        static void setEndTemperature(const double endTemperature)
        {
            _endTemperature = endTemperature;
            setEndTemperatureSet(true);
        }

        static void setRelaxationTime(const double relaxationTime)
        {
            _relaxationTime = relaxationTime;
        }

        static void setFriction(const double friction) { _friction = friction; }

        static void setNoseHooverCouplingFrequency(const double frequency)
        {
            _noseHooverCouplingFrequency = frequency;
        }

        static auto addChi(const size_t index, const double chi)
        {
            return _chi.try_emplace(index, chi);
        }

        static auto addZeta(const size_t index, const double zeta)
        {
            return _zeta.try_emplace(index, zeta);
        }

        /***************************
         * standard getter methods *
         ***************************/

        // size_t getters

        [[nodiscard]] static size_t getNoseHooverChainLength()
        {
            return _noseHooverChainLength;
        }

        [[nodiscard]] static size_t getTemperatureRampSteps()
        {
            return _temperatureRampSteps;
        }

        [[nodiscard]] static size_t getTemperatureRampFrequency()
        {
            return _temperatureRampFrequency;
        }

        // ThermostatType getters

        [[nodiscard]] static ThermostatType getThermostatType()
        {
            return _thermostatType;
        }

        // bool getters

        [[nodiscard]] static bool isTemperatureSet()
        {
            return _isTemperatureSet;
        }

        [[nodiscard]] static bool isStartTemperatureSet()
        {
            return _isStartTemperatureSet;
        }

        [[nodiscard]] static bool isEndTemperatureSet()
        {
            return _isEndTemperatureSet;
        }

        // double getters

        [[nodiscard]] static double getTargetTemperature()
        {
            return _targetTemperature;
        }

        [[nodiscard]] static double getActualTargetTemperature()
        {
            return _actualTargetTemperature;
        }

        [[nodiscard]] static double getStartTemperature()
        {
            return _startTemperature;
        }

        [[nodiscard]] static double getEndTemperature()
        {
            return _endTemperature;
        }

        [[nodiscard]] static double getRelaxationTime()
        {
            return _relaxationTime;
        }

        [[nodiscard]] static double getFriction() { return _friction; }

        [[nodiscard]] static double getNoseHooverCouplingFrequency()
        {
            return _noseHooverCouplingFrequency;
        }

        [[nodiscard]] static std::map<size_t, double> getChi() { return _chi; }

        [[nodiscard]] static std::map<size_t, double> getZeta()
        {
            return _zeta;
        }
    };
}   // namespace settings

#endif   // _THERMOSTAT_SETTINGS_HPP_