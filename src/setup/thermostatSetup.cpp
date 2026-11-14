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

#include <format>
#include <string>

#include "berendsenThermostat.hpp"
#include "langevinThermostat.hpp"
#include "mdEngine.hpp"
#include "noseHooverThermostat.hpp"
#include "thermostat.hpp"
#include "thermostatSettings.hpp"
#include "units.hpp"
#include "velocityRescalingThermostat.hpp"

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
            _engine.getSettings().thermostatSettings.getThermostatType();

        switch (thermostatType)
        {
            case BERENDSEN: setupBerendsenThermostat(); break;

            case VELOCITY_RESCALING: setupVelocityRescalingThermostat(); break;

            case LANGEVIN: setupLangevinThermostat(); break;

            case NOSE_HOOVER: setupNoseHooverThermostat(); break;

            case NONE: _engine.makeThermostat(thermostat::Thermostat());
        }

        writeSetupInfo();
    }

    /**
     * @brief setup berendsen thermostat
     *
     * @details constructs a berendsen thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupBerendsenThermostat()
    {
        const auto tau = ThermostatSettings::getRelaxationTime();

        _engine.makeThermostat(thermostat::BerendsenThermostat(tau));
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
        const auto tau = ThermostatSettings::getRelaxationTime();

        _engine.makeThermostat(thermostat::VelocityRescalingThermostat(tau));
    }

    /**
     * @brief setup langevin thermostat
     *
     * @details constructs a langevin thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupLangevinThermostat()
    {
        const auto friction = ThermostatSettings::getFriction();

        _engine.makeThermostat(thermostat::LangevinThermostat(friction));
    }

    /**
     * @brief setup nose hoover thermostat
     *
     * @details constructs a nose hoover thermostat and adds it to the engine
     *
     */
    void ThermostatSetup::setupNoseHooverThermostat()
    {
        const auto &settings = _engine.getSettings().noseHooverSettings;

        auto thermostat = thermostat::NoseHooverThermostat(settings);

        _engine.makeThermostat(thermostat);
    }

    void ThermostatSetup::writeSetupInfo() const
    {
        auto      &log = _engine.getLogOutput();
        const auto thermostatType =
            _engine.getSettings().thermostatSettings.getThermostatType();

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

        if (_engine.getSettings().kinetics.isStartTemperatureSet())
            writeTemperatureRampInfo();
    }

    /**
     * @brief write berendsen thermostat info
     *
     */
    void ThermostatSetup::writeBerendsenInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto &kineticsSettings = _engine.getSettings().kinetics;
        const auto  targetTemp       = kineticsSettings.getEndTemperature();
        const auto  tau              = ThermostatSettings::getRelaxationTime();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(std::format("Relaxation time:    {}", tau.in(ps)));
        log.writeEmptyLine();
    }

    /**
     * @brief write langevin thermostat info
     *
     */
    void ThermostatSetup::writeLangevinInfo() const
    {
        auto &log = _engine.getLogOutput();

        const auto &kineticsSettings = _engine.getSettings().kinetics;
        const auto  targetTemp       = kineticsSettings.getEndTemperature();
        const auto  friction         = ThermostatSettings::getFriction();

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

        const auto &kineticsSettings = _engine.getSettings().kinetics;
        const auto  targetTemp       = kineticsSettings.getEndTemperature();
        const auto &nhSettings       = _engine.getSettings().noseHooverSettings;
        const auto  nhChainLength    = nhSettings.getNoseHooverChainLength();
        const auto  couplFreq = nhSettings.getNoseHooverCouplingFrequency();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(
            std::format("NH chain length:    {}", nhChainLength)
        );
        log.writeSetupInfo(
            std::format(
                "NH coupling freq:   {}",
                mstd::toWavenumber(couplFreq).in(per_cm)
            )
        );
        log.writeEmptyLine();
    }

    /**
     * @brief write temperature ramp info
     *
     */
    void ThermostatSetup::writeTemperatureRampInfo() const
    {
        auto       &log              = _engine.getLogOutput();
        const auto &kineticsSettings = _engine.getSettings().kinetics;

        const auto startTemp  = kineticsSettings.getStartTemperature();
        const auto targetTemp = kineticsSettings.getEndTemperature();
        const auto steps      = kineticsSettings.getRampingSteps();
        const auto frequency  = kineticsSettings.getRampingFrequency();
        const auto tempStep   = kineticsSettings.getTemperatureIncrease();

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
        auto       &log      = _engine.getLogOutput();
        const auto &settings = _engine.getSettings();

        const auto targetTemp = settings.kinetics.getEndTemperature();
        const auto tau        = ThermostatSettings::getRelaxationTime();

        log.writeSetupInfo(std::format("Target temperature: {} K", targetTemp));
        log.writeSetupInfo(std::format("Relaxation time:    {}", tau.in(ps)));
        log.writeEmptyLine();
    }

    /**
     * @brief get the engine
     *
     * @return const ThermostatSetup::MDEngine&
     */
    engine::MDEngine &ThermostatSetup::getEngine() const { return _engine; }

}   // namespace setup
