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

#include "thermostatSetup.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for identity
#include <map>          // for map, operator==
#include <string>       // for string
#include <vector>       // for vector

#include "berendsenThermostat.hpp"           // for BerendsenThermostat
#include "constants/conversionFactors.hpp"   // for _PS_TO_FS_, _PER_CM_TO_HZ_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for InputFileException
#include "langevinThermostat.hpp"            // for LangevinThermostat
#include "noseHooverThermostat.hpp"          // for NoseHooverThermostat
#include "thermostat.hpp"                    // for Thermostat
#include "thermostatSettings.hpp"   // for ThermostatSettings, ThermostatType
#include "timingsSettings.hpp"      // for TimingsSettings
#include "velocityRescalingThermostat.hpp"   // for VelocityRescalingThermostat

using namespace setup;
using namespace settings;

/**
 * @brief wrapper for thermostat setup
 *
 * @details constructs a thermostat setup object and calls the setup function
 *
 * @param engine
 */
void setup::setupThermostat(engine::Engine &engine)
{
    engine.getStdoutOutput().writeSetup("thermostat");
    engine.getLogOutput().writeSetup("thermostat");

    ThermostatSetup thermostatSetup(engine);
    thermostatSetup.setup();
}

/**
 * @brief setup thermostat
 *
 * @details checks if a thermostat was set in the input file,
 * If a thermostat was selected than the user has to provide a target
 * temperature for the thermostat.
 *
 * @note the base class Thermostat does not apply any temperature coupling to
 * the system and therefore it represents the none thermostat.
 *
 * @throws InputFileException if no temperature was set for the thermostat
 *
 */
void ThermostatSetup::setup()
{
    const auto thermostatType = ThermostatSettings::getThermostatType();

    if (thermostatType != ThermostatType::NONE)
        isTargetTemperatureSet();

    if (thermostatType == ThermostatType::BERENDSEN)
        setupBerendsenThermostat();

    else if (thermostatType == ThermostatType::VELOCITY_RESCALING)
        setupVelocityRescalingThermostat();

    else if (thermostatType == ThermostatType::LANGEVIN)
        setupLangevinThermostat();

    else if (thermostatType == ThermostatType::NOSE_HOOVER)
        setupNoseHooverThermostat();

    else
    {
        _engine.makeThermostat(thermostat::Thermostat());
        _engine.getLogOutput().writeSetupInfo("No thermostat selected");
    }

    _engine.getLogOutput().writeEmptyLine();

    setupTemperatureRamp();
}

/**
 * @brief check if target temperature is set
 *
 * @throws InputFileException if neither target nor end temperature is set
 * @throws InputFileException if both target and end temperature are set
 *
 */
void ThermostatSetup::isTargetTemperatureSet() const
{
    auto targetTempDefined = ThermostatSettings::isTemperatureSet();
    auto endTempDefined    = ThermostatSettings::isEndTemperatureSet();

    /************************************************************
     * Check if exactly one of target or end temperature is set *
     ************************************************************/

    if (!targetTempDefined && !endTempDefined)
        throw customException::InputFileException(std::format(
            "Target or end temperature not set for {} thermostat",
            string(ThermostatSettings::getThermostatType())
        ));

    if (targetTempDefined && endTempDefined)
        throw customException::InputFileException(std::format(
            "Both target and end temperature set for {} thermostat. They are "
            "mutually exclusive as they are treated as synonyms",
            string(ThermostatSettings::getThermostatType())
        ));

    /**************************************************
     * Block to unify the target and end temperature. *
     **************************************************/

    if (endTempDefined)
        ThermostatSettings::setTargetTemperature(
            ThermostatSettings::getEndTemperature()
        );

    if (targetTempDefined)
        ThermostatSettings::setEndTemperature(
            ThermostatSettings::getTargetTemperature()
        );

    /******************************************
     * Writing Target Temperature to Log File *
     ******************************************/

    _engine.getLogOutput().writeSetupInfo(std::format(
        "target temperature: {:14.5f} K",
        ThermostatSettings::getTargetTemperature()
    ));
    _engine.getLogOutput().writeEmptyLine();
}

/**
 * @brief setup berendsen thermostat
 *
 * @details constructs a berendsen thermostat and adds it to the engine
 *
 */
void ThermostatSetup::setupBerendsenThermostat()
{
    _engine.makeThermostat(thermostat::BerendsenThermostat(
        ThermostatSettings::getTargetTemperature(),
        ThermostatSettings::getRelaxationTime() * constants::_PS_TO_FS_
    ));

    _engine.getLogOutput().writeSetupInfo("Berendsen thermostat:");
    _engine.getLogOutput().writeSetupInfo(std::format(
        "relaxation time: {:14.5f} ps",
        ThermostatSettings::getRelaxationTime()
    ));
}

/**
 * @brief setup velocity rescaling thermostat
 *
 * @details constructs a velocity rescaling thermostat and adds it to the engine
 *
 */
void ThermostatSetup::setupVelocityRescalingThermostat()
{
    _engine.makeThermostat(thermostat::VelocityRescalingThermostat(
        ThermostatSettings::getTargetTemperature(),
        ThermostatSettings::getRelaxationTime() * constants::_PS_TO_FS_
    ));

    _engine.getLogOutput().writeSetupInfo("Velocity Rescaling thermostat:");
    _engine.getLogOutput().writeSetupInfo(std::format(
        "relaxation time: {:14.5f} ps",
        ThermostatSettings::getRelaxationTime()
    ));
}

/**
 * @brief setup langevin thermostat
 *
 * @details constructs a langevin thermostat and adds it to the engine
 *
 */
void ThermostatSetup::setupLangevinThermostat()
{
    _engine.makeThermostat(thermostat::LangevinThermostat(
        ThermostatSettings::getTargetTemperature(),
        ThermostatSettings::getFriction()
    ));

    _engine.getLogOutput().writeSetupInfo("Langevin thermostat:");
    _engine.getLogOutput().writeSetupInfo(std::format(
        "friction: {:14.5f} ps⁻¹",
        ThermostatSettings::getFriction()
    ));
}

/**
 * @brief setup nose hoover thermostat
 *
 * @details constructs a nose hoover thermostat and adds it to the engine
 *
 */
void ThermostatSetup::setupNoseHooverThermostat()
{
    const auto noseHooverChainLength =
        ThermostatSettings::getNoseHooverChainLength();
    const auto noseHooverCouplingFrequency =
        ThermostatSettings::getNoseHooverCouplingFrequency() *
        constants::_PER_CM_TO_HZ_;

    auto thermostat = thermostat::NoseHooverThermostat(
        ThermostatSettings::getTargetTemperature(),
        std::vector<double>(noseHooverChainLength + 1, 0.0),
        std::vector<double>(noseHooverChainLength + 1, 0.0),
        noseHooverCouplingFrequency
    );

    auto fillChi = [&thermostat, noseHooverChainLength](const auto pair)
    {
        if (pair.first > noseHooverChainLength)
            throw customException::InputFileException(std::format(
                "Chi index {} is larger than the number of nose hoover chains "
                "{}",
                pair.first,
                noseHooverChainLength
            ));

        thermostat.setChi(size_t(pair.first - 1), pair.second);
    };

    auto fillZeta = [&thermostat](const auto pair)
    { thermostat.setZeta(size_t(pair.first - 1), pair.second); };

    std::ranges::for_each(ThermostatSettings::getChi(), fillChi);
    std::ranges::for_each(ThermostatSettings::getZeta(), fillZeta);

    _engine.makeThermostat(thermostat);

    _engine.getLogOutput().writeSetupInfo("Nose-Hoover chain thermostat:");
    _engine.getLogOutput().writeSetupInfo(std::format(
        "chain length:       {:8d}",
        ThermostatSettings::getNoseHooverChainLength()
    ));
    _engine.getLogOutput().writeSetupInfo(std::format(
        "coupling frequency: {:14.5f} cm⁻¹",
        ThermostatSettings::getNoseHooverCouplingFrequency()
    ));
}

/**
 * @brief setup temperature ramp
 *
 * @details if the start temperature is defined, the temperature ramp is enabled
 *
 * @throws InputFileException if the number of steps is smaller than the number
 * @throws InputFileException if the temperature ramp frequency is larger than
 * the number of steps
 *
 */
void ThermostatSetup::setupTemperatureRamp()
{
    /*************************************************************************
     * If the start temperature is defined, the temperature ramp is enabled. *
     *************************************************************************/

    if (!ThermostatSettings::isStartTemperatureSet())
        return;

    /*************************************************************
     * resetting the target temperature to the start temperature *
     *************************************************************/

    _engine.getThermostat().setTargetTemperature(
        ThermostatSettings::getStartTemperature()
    );

    auto steps = ThermostatSettings::getTemperatureRampSteps();

    /*************************************************************
     * If steps is 0, set the steps to the total number of steps *
     *************************************************************/

    if (steps == 0)
    {
        steps = TimingsSettings::getNumberOfSteps();
        ThermostatSettings::setTemperatureRampSteps(steps);
    }
    else if (steps > TimingsSettings::getNumberOfSteps())
        throw customException::InputFileException(std::format(
            "Number of total simulation steps {} is smaller than the number of "
            "temperature ramping steps {}",
            TimingsSettings::getNumberOfSteps(),
            steps
        ));

    _engine.getThermostat().setTemperatureRampingSteps(steps);

    const auto frequency = ThermostatSettings::getTemperatureRampFrequency();

    if (frequency > steps)
        throw customException::InputFileException(std::format(
            "Temperature ramp frequency {} is larger than the number of steps "
            "{}",
            frequency,
            steps
        ));

    const auto temperatureIncrease =
        (ThermostatSettings::getTargetTemperature() -
         ThermostatSettings::getStartTemperature()) /
        double(steps);

    _engine.getThermostat().setTemperatureIncrease(temperatureIncrease);

    /****************************************************
     * Writing Temperature Ramp Information to Log File *
     ****************************************************/

    _engine.getLogOutput().writeEmptyLine();
    _engine.getLogOutput().writeSetupInfo("Temperature Ramp enabled:");
    _engine.getLogOutput().writeSetupInfo(std::format(
        "start temperature:  {:14.5f} K",
        ThermostatSettings::getStartTemperature()
    ));
    _engine.getLogOutput().writeSetupInfo(std::format(
        "end temperature:    {:14.5f} K",
        ThermostatSettings::getTargetTemperature()
    ));
    _engine.getLogOutput().writeSetupInfo(
        std::format("ramping steps:      {:8d}", steps)
    );
    _engine.getLogOutput().writeSetupInfo(
        std::format("ramping frequency:  {:8d}", frequency)
    );
    _engine.getLogOutput().writeSetupInfo(
        std::format("temperature increase: {:10.5f} K", temperatureIncrease)
    );
    _engine.getLogOutput().writeEmptyLine();
}