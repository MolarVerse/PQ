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

#ifndef SETTINGS__THERMOSTAT_SETTINGS_HPP
#define SETTINGS__THERMOSTAT_SETTINGS_HPP

#include <cstddef>
#include <map>
#include <mstd/enum.hpp>

#include "defaults.hpp"
#include "enums/thermostat.hpp"
#include "setting.hpp"
#include "unit_types.hpp"

class ReferenceSettings;

/**
 * @class ThermostatSettings
 *
 * @brief static class to store settings of the thermostat
 *
 */
class ThermostatSettings
{
   private:
    Setting<ThermostatType, "Thermostat Type"> _thermostatType{
        ThermostatType::NONE
    };

    static inline bool _isTemperatureSet      = false;
    static inline bool _isStartTemperatureSet = false;
    static inline bool _isEndTemperatureSet   = false;

    static inline size_t _nhChainLength = defaults::NH_CHAIN_LENGTH_DEFAULT;
    static inline size_t _temperatureRampSteps     = 0;
    static inline size_t _temperatureRampFrequency = 1;

    static inline Temperature _targetTemperature;
    static inline Temperature _actualTargetTemperature;
    static inline Temperature _startTemperature;
    static inline Temperature _endTemperature;

    static inline Time _relaxationTime{
        defaults::BERENDSEN_THERMOSTAT_RELAX_TIME
    };
    static inline double _friction{defaults::LANGEVIN_THERMOSTAT_FRICTION};
    static inline double _nhCouplingFreq{defaults::NH_COUPLING_FREQ};

    static inline std::map<size_t, double> _chi;
    static inline std::map<size_t, double> _zeta;

   public:
    void finalize(ReferenceSettings& referenceSettings);

    static auto addChi(size_t index, double chi)
        -> decltype(_chi.try_emplace(index, chi));
    static auto addZeta(size_t index, double zeta)
        -> decltype(_zeta.try_emplace(index, zeta));

    /***************************
     * standard setter methods *
     ***************************/

    void setThermostatTypeByUser(ThermostatType thermostatType);

    static void setTemperatureSet(bool);
    static void setStartTemperatureSet(bool);
    static void setEndTemperatureSet(bool);
    static void setTargetTemperature(const Temperature&);
    static void setActualTargetTemperature(const Temperature&);
    static void setStartTemperature(const Temperature&);
    static void setEndTemperature(const Temperature&);

    static void setTemperatureRampSteps(size_t);
    static void setTemperatureRampFrequency(size_t);

    static void setRelaxationTime(const Time& relaxationTime);
    static void setFriction(double);
    static void setNoseHooverChainLength(size_t);
    static void setNoseHooverCouplingFrequency(double);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] ThermostatType getThermostatType() const;

    [[nodiscard]] static size_t getNoseHooverChainLength();
    [[nodiscard]] static size_t getTemperatureRampSteps();
    [[nodiscard]] static size_t getTemperatureRampFrequency();

    [[nodiscard]] static bool isTemperatureSet();
    [[nodiscard]] static bool isStartTemperatureSet();
    [[nodiscard]] static bool isEndTemperatureSet();

    [[nodiscard]] static const Temperature& getTargetTemperature();
    [[nodiscard]] static const Temperature& getActualTargetTemperature();
    [[nodiscard]] static const Temperature& getStartTemperature();
    [[nodiscard]] static const Temperature& getEndTemperature();
    [[nodiscard]] static const Time&        getRelaxationTime();
    [[nodiscard]] static double             getFriction();
    [[nodiscard]] static double             getNoseHooverCouplingFrequency();

    [[nodiscard]] static std::map<size_t, double> getChi();
    [[nodiscard]] static std::map<size_t, double> getZeta();
};

#endif   // SETTINGS__THERMOSTAT_SETTINGS_HPP
