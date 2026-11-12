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
 * @class NoseHooverSettings
 *
 * @brief Stores settings specific to the Nose-Hoover thermostat
 *
 */
class NoseHooverSettings
{
   private:
    Setting<size_t, "Nose-Hoover Chain Length"> _nhChainLength{
        defaults::NH_CHAIN_LENGTH_DEFAULT
    };
    Setting<Frequency, "Nose-Hoover Coupling Frequency"> _nhCouplingFreq{
        defaults::NH_COUPLING_FREQ
    };

    Setting<std::map<size_t, double>, "Nose-Hoover Chi Values">  _chi;
    Setting<std::map<size_t, double>, "Nose-Hoover Zeta Values"> _zeta;

   public:
    void setNoseHooverChainLengthByUser(size_t chainLength);
    void setNoseHooverCouplingFrequencyByUser(const Frequency& frequency);

    [[nodiscard]] size_t           getNoseHooverChainLength() const;
    [[nodiscard]] const Frequency& getNoseHooverCouplingFrequency() const;

    void setChiByUser(const std::map<size_t, double>& chi);
    void setZetaByUser(const std::map<size_t, double>& zeta);

    [[nodiscard]] const std::map<size_t, double>& getChi() const;
    [[nodiscard]] const std::map<size_t, double>& getZeta() const;

    [[nodiscard]] bool isChiAndZetaSet() const;
};

/**
 * @class KineticsSettings
 *
 * @brief Stores settings specific to the kinetics
 *
 */
class KineticsSettings
{
   private:
    Setting<size_t, "Temperature Ramp Frequency"> _temperatureRampFrequency{1};
    Setting<size_t, "Temperature Ramp Steps">     _temperatureRampSteps{0};
    Setting<Temperature, "Start Temperature">     _startTemperature;
    Setting<Temperature, "End Temperature">       _endTemperature;
    Setting<Temperature, "Target Temperature">    _targetTemperature;
    Setting<TemperatureDiff, "Temperature Increase"> _temperatureIncrease;

   public:
    void finalize();

    [[nodiscard]]
    size_t getRampingSteps() const;
    [[nodiscard]]
    size_t getRampingFrequency() const;
    [[nodiscard]]
    const TemperatureDiff& getTemperatureIncrease() const;
    [[nodiscard]]
    const Temperature& getStartTemperature() const;
    [[nodiscard]]
    const Temperature& getEndTemperature() const;
    [[nodiscard]]
    bool isStartTemperatureSet() const;

    void setRampingFrequencyByUser(size_t frequency);
    void setRampingStepsByUser(size_t steps);
    void setStartTemperatureByUser(const Temperature& startTemperature);
    void setEndTemperatureByUser(const Temperature& endTemperature);
    void setTargetTemperatureByUser(const Temperature& targetTemperature);
};

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

    static inline Time _relaxationTime{
        defaults::BERENDSEN_THERMOSTAT_RELAX_TIME
    };
    static inline double _friction{defaults::LANGEVIN_THERMOSTAT_FRICTION};

   public:
    void finalize(ReferenceSettings& referenceSettings);

    /***************************
     * standard setter methods *
     ***************************/

    void setThermostatTypeByUser(ThermostatType thermostatType);

    static void setRelaxationTime(const Time& relaxationTime);
    static void setFriction(double);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] ThermostatType getThermostatType() const;

    [[nodiscard]] static const Time& getRelaxationTime();
    [[nodiscard]] static double      getFriction();
};

#endif   // SETTINGS__THERMOSTAT_SETTINGS_HPP
