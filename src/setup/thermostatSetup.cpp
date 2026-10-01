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

#include <algorithm>   // for __for_each_fn, for_each
#include <cstddef>     // for size_t
#include <format>      // for format
#include <string>      // for string
#include <vector>      // for vector

#include "berendsenThermostat.hpp"           // for BerendsenThermostat
#include "constants/conversionFactors.hpp"   // for _PS_TO_FS_, _PER_CM_TO_HZ_
#include "exceptions.hpp"                    // for InputFileException
#include "langevinThermostat.hpp"            // for LangevinThermostat
#include "mdEngine.hpp"                      // for Engine
#include "noseHooverThermostat.hpp"          // for NoseHooverThermostat
#include "thermostat.hpp"                    // for Thermostat
#include "thermostatSettings.hpp"   // for settings::ThermostatSettings, ThermostatType
#include "timingsSettings.hpp"               // for settings::TimingsSettings
#include "velocityRescalingThermostat.hpp"   // for VelocityRescalingThermostat

namespace setup
{

    /**
     * @brief wrapper for thermostat setup
     *
     * @details constructs a thermostat setup object and calls the setup
     * function
     *
     * @param engine
     */
    void setupThermostat(engine::Engine &engine)
    {
        out::StdoutOutput::writeSetup("thermostat");
        engine.getLogOutput().writeSetup("thermostat");

        ThermostatSetup thermostatSetup(
            dynamic_cast<engine::MDEngine &>(engine)
        );
        thermostatSetup.setup();
    }

    /**
     * @brief Construct a new Thermostat Setup object
     *
     * @param engine
     */
    ThermostatSetup::ThermostatSetup(engine::MDEngine &engine) : _engine(engine)
    {
    }

    /**
     * @brief setup thermostat
     *
     * @note the base class Thermostat does not apply any temperature coupling
     * to the system and therefore it represents the none thermostat.
     */
    void ThermostatSetup::setup()
    {
        using enum ThermostatType;

        const auto thermostatType =
            settings::ThermostatSettings::getThermostatType();

        if (thermostatType != NONE)
            setupTargetTemperature();

        switch (thermostatType)
        {
            case BERENDSEN: setupBerendsenThermostat(); break;

            case VELOCITY_RESCALING: setupVelocityRescalingThermostat(); break;

            case LANGEVIN: setupLangevinThermostat(); break;

            case NOSE_HOOVER: setupNoseHooverThermostat(); break;

            case NONE: _engine.makeThermostat(thermostat::Thermostat());
        }

        setupTemperatureRamp();

        writeSetupInfo();
    }

    /**
     * @brief keeps target and end temperature synchronized
     */
    void ThermostatSetup::setupTargetTemperature()
    {
        const auto targetTempDefined =
            settings::ThermostatSettings::isTemperatureSet();
        const auto endTempDefined =
            settings::ThermostatSettings::isEndTemperatureSet();

        if (endTempDefined)
        {
            const auto endTemp =
                settings::ThermostatSettings::getEndTemperature();
            settings::ThermostatSettings::setTargetTemperature(endTemp);
        }

        if (targetTempDefined)
        {
            const auto targetTemp =
                settings::ThermostatSettings::getTargetTemperature();
            settings::ThermostatSettings::setEndTemperature(targetTemp);
        }
    }

    /**
     * @brief setup berendsen thermostat
     *
     * @details constructs a berendsen thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupBerendsenThermostat()
    {
        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto tau =
            settings::ThermostatSettings::getRelaxationTime() * PS_TO_FS;

        _engine.makeThermostat(
            thermostat::BerendsenThermostat(targetTemp, tau)
        );
    }

    /**
     * @brief setup velocity rescaling thermostat
     *
     * @details constructs a velocity rescaling thermostat and adds it to the
     * engine
     *
     */
    void ThermostatSetup::setupVelocityRescalingThermostat()
    {
        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto tau =
            settings::ThermostatSettings::getRelaxationTime() * PS_TO_FS;

        _engine.makeThermostat(
            thermostat::VelocityRescalingThermostat(targetTemp, tau)
        );
    }

    /**
     * @brief setup langevin thermostat
     *
     * @details constructs a langevin thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupLangevinThermostat()
    {
        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto friction = settings::ThermostatSettings::getFriction();

        _engine.makeThermostat(
            thermostat::LangevinThermostat(targetTemp, friction)
        );
    }

    /**
     * @brief setup nose hoover thermostat
     *
     * @details constructs a nose hoover thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupNoseHooverThermostat()
    {
        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto nhChainLength =
            settings::ThermostatSettings::getNoseHooverChainLength();

        auto nhCouplFreq =
            settings::ThermostatSettings::getNoseHooverCouplingFrequency();
        nhCouplFreq *= PER_CM_TO_HZ;

        const auto chi  = std::vector<double>(nhChainLength + 1, 0.0);
        const auto zeta = std::vector<double>(nhChainLength + 1, 0.0);

        auto thermostat = thermostat::NoseHooverThermostat(
            targetTemp,
            chi,
            zeta,
            nhCouplFreq
        );

        auto fillChi = [&thermostat, nhChainLength](const auto pair)
        {
            if (pair.first > nhChainLength)
            {
                throw exc::InputFileException(
                    std::format(
                        "Chi index {} is larger than the number of nose hoover "
                        "chains {}",
                        pair.first,
                        nhChainLength
                    )
                );
            }

            thermostat.setChi(size_t(pair.first - 1), pair.second);
        };

        auto fillZeta = [&thermostat](const auto pair)
        { thermostat.setZeta(size_t(pair.first - 1), pair.second); };

        std::ranges::for_each(settings::ThermostatSettings::getChi(), fillChi);
        std::ranges::for_each(
            settings::ThermostatSettings::getZeta(),
            fillZeta
        );

        _engine.makeThermostat(thermostat);
    }

    /**
     * @brief setup temperature ramp
     *
     * @details if the start temperature is defined, the temperature ramp is
     * enabled
     *
     */
    void ThermostatSetup::setupTemperatureRamp()
    {
        /*************************************************************************
         * If the start temperature is defined, the temperature ramp is enabled.
         **
         *************************************************************************/

        if (!settings::ThermostatSettings::isStartTemperatureSet())
            return;

        /*************************************************************
         * If steps is 0, set the steps to the total number of steps *
         *************************************************************/

        auto steps = settings::ThermostatSettings::getTemperatureRampSteps();
        const auto useFullSimulation = steps == 0;

        if (useFullSimulation)
            steps = settings::TimingsSettings::getNumberOfSteps();

        if (steps == 0)
            throw exc::InputFileException(
                "Temperature ramp requires at least one simulation step"
            );

        const auto frequency =
            settings::ThermostatSettings::getTemperatureRampFrequency();

        if (frequency == 0)
            throw exc::InputFileException(
                "Temperature ramp frequency must be greater than zero"
            );

        if (useFullSimulation)
            settings::ThermostatSettings::setTemperatureRampSteps(steps);

        /*************************************************************
         * resetting the target temperature to the start temperature *
         *************************************************************/

        const auto startTemp =
            settings::ThermostatSettings::getStartTemperature();

        _engine.getThermostat().setTargetTemperature(startTemp);
        settings::ThermostatSettings::setActualTargetTemperature(startTemp);
        _engine.getThermostat().setTemperatureRampingSteps(steps);
        _engine.getThermostat().setTemperatureRampingFrequency(frequency);

        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto tempDelta    = targetTemp - startTemp;
        const auto remainder    = steps % frequency == 0 ? 0U : 1U;
        const auto updates      = (steps / frequency) + remainder;
        const auto tempIncrease = tempDelta / static_cast<double>(updates);

        _engine.getThermostat().setTemperatureIncrease(tempIncrease);
    }

    void ThermostatSetup::writeSetupInfo() const
    {
        auto      &log = _engine.getLogOutput();
        const auto thermostatType =
            settings::ThermostatSettings::getThermostatType();

        if (thermostatType == ThermostatType::NONE)
            log.writeSetupInfo("No thermostat selected");
        else
        {
            log.writeSetupInfo(
                std::format(
                    "Thermostat type: {}",
                    ThermostatTypeMeta::toString(thermostatType)
                )
            );
            log.writeEmptyLine();
        }

        if (thermostatType == ThermostatType::BERENDSEN)
            writeBerendsenInfo();

        else if (thermostatType == ThermostatType::VELOCITY_RESCALING)
            writeVelocityRescalingInfo();

        else if (thermostatType == ThermostatType::LANGEVIN)
            writeLangevinInfo();

        else if (thermostatType == ThermostatType::NOSE_HOOVER)
            writeNoseHooverInfo();

        if (settings::ThermostatSettings::isStartTemperatureSet())
            writeTemperatureRampInfo();
    }

    /**
     * @brief write berendsen thermostat info
     *
     */
    void ThermostatSetup::writeBerendsenInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto tau = settings::ThermostatSettings::getRelaxationTime();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(std::format("Relaxation time:    {} ps", tau));
        log.writeEmptyLine();
    }

    /**
     * @brief write langevin thermostat info
     *
     */
    void ThermostatSetup::writeLangevinInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto friction = settings::ThermostatSettings::getFriction();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(
            std::format("Friction:           {} 1/ps", friction)
        );
        log.writeEmptyLine();
    }

    /**
     * @brief write nose hoover thermostat info
     *
     */
    void ThermostatSetup::writeNoseHooverInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto nhChainLength =
            settings::ThermostatSettings::getNoseHooverChainLength();
        const auto couplFreq =
            settings::ThermostatSettings::getNoseHooverCouplingFrequency();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(
            std::format("NH chain length:    {}", nhChainLength)
        );
        log.writeSetupInfo(
            std::format("NH coupling freq:   {} cm⁻¹", couplFreq)
        );
        log.writeEmptyLine();
    }

    /**
     * @brief write temperature ramp info
     *
     */
    void ThermostatSetup::writeTemperatureRampInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto startTemp =
            settings::ThermostatSettings::getStartTemperature();
        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto steps =
            settings::ThermostatSettings::getTemperatureRampSteps();
        const auto frequency =
            settings::ThermostatSettings::getTemperatureRampFrequency();
        const auto tempStep = _engine.getThermostat().getTemperatureIncrease();

        log.writeSetupInfo(std::format("Start temp:          {} K", startTemp));
        log.writeSetupInfo(
            std::format("Target temp:         {} K", targetTemp)
        );
        log.writeSetupInfo(std::format("Temp ramp increase:  {} K", tempStep));
        log.writeSetupInfo(std::format("Temp ramp steps:     {}", steps));
        log.writeSetupInfo(std::format("Temp ramp frequency: {}", frequency));
        log.writeEmptyLine();
    }

    /**
     * @brief write velocity rescaling thermostat info
     *
     */
    void ThermostatSetup::writeVelocityRescalingInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto targetTemp =
            settings::ThermostatSettings::getTargetTemperature();
        const auto tau = settings::ThermostatSettings::getRelaxationTime();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(std::format("Relaxation time:    {} ps", tau));
        log.writeEmptyLine();
    }

    /**
     * @brief get the engine
     *
     * @return const ThermostatSetup::MDEngine&
     */
    engine::MDEngine &ThermostatSetup::getEngine() const { return _engine; }

}   // namespace setup
