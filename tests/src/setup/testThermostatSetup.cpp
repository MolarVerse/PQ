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

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <string>

#include "berendsenThermostat.hpp"
#include "engineOutput.hpp"
#include "logOutput.hpp"
#include "noseHooverThermostat.hpp"
#include "testSetup.hpp"
#include "thermostatSettings.hpp"
#include "thermostatSetup.hpp"
#include "timingsSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "velocityRescalingThermostat.hpp"

namespace
{
    // the setup logs the target temperature, which is resolved in finalize()
    void finalizeTargetTemperature(Settings &settings)
    {
        settings.kinetics.setTargetTemperatureByUser(toTemperature(300.0));
        settings.timings.setNumberOfStepsByUser(100U);
        settings.kinetics.finalize(settings.timings.getNumberOfSteps());
    }
}   // namespace

TEST_F(TestSetup, setupThermostatNoThermostat)
{
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    _settings->timings.setTimeStepByUser(0.1 * fs);
    EXPECT_NO_THROW(thermostatSetup.setup());
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getThermostatType(),
        ThermostatType::NONE
    );
}

TEST_F(TestSetup, setupThermostatBerendsen)
{
    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::BERENDSEN
    );
    finalizeTargetTemperature(*_settings);
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto &berendsenThermostat =
        dynamic_cast<thermostat::BerendsenThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(berendsenThermostat.getTau(), 0.1 * 1000 * fs);

    _settings->thermostatSettings.setRelaxationTimeByUser(0.2 * ps);

    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto &berendsenThermostat2 =
        dynamic_cast<thermostat::BerendsenThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(berendsenThermostat2.getTau(), 0.2 * 1000 * fs);
}

TEST_F(TestSetup, setupThermostatVelocityRescaling)
{
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::VELOCITY_RESCALING
    );
    finalizeTargetTemperature(*_settings);
    _settings->thermostatSettings.setRelaxationTimeByUser(0.2 * ps);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto &velocityRescalingThermostat =
        dynamic_cast<thermostat::VelocityRescalingThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(velocityRescalingThermostat.getTau(), 0.2 * 1000 * fs);
}

TEST_F(TestSetup, setupThermostatLangevin)
{
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::LANGEVIN
    );
    finalizeTargetTemperature(*_settings);
    EXPECT_NO_THROW(thermostatSetup.setup());

    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getThermostatType(),
        ThermostatType::LANGEVIN
    );
    EXPECT_EQ(_settings->thermostatSettings.getFriction(), 1.0e11 * Hz);

    EXPECT_NO_THROW(setup::setupThermostat(*_mdEngine));
}

TEST_F(TestSetup, setupThermostatNhChain)
{
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::NOSE_HOOVER
    );
    finalizeTargetTemperature(*_settings);
    _settings->noseHooverSettings.setNoseHooverChainLengthByUser(5);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto &noseHooverThermostat =
        dynamic_cast<thermostat::NoseHooverThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(noseHooverThermostat.getChi().size(), 5);
    EXPECT_EQ(noseHooverThermostat.getZeta().size(), 5);
}

namespace
{
    /**
     * @brief runs a function that writes to the log of the engine and returns
     * what it wrote
     */
    template <typename Writer>
    std::string readLogWrittenBy(engine::MDEngine &engine, Writer write)
    {
        const std::string fileName = "thermostatSetup.log";

        engine.getEngineOutput().getLogOutput().setFilename(fileName, true);
        write();
        engine.getEngineOutput().getLogOutput().close();

        std::ifstream log(fileName);
        return {
            std::istreambuf_iterator<char>(log),
            std::istreambuf_iterator<char>()
        };
    }
}   // namespace

/**
 * @brief the temperature is printed once with its unit, not as "300 K K"
 */
TEST_F(TestSetup, thermostatLogPrintsTheTemperatureUnitOnce)
{
    finalizeTargetTemperature(*_settings);
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    const auto text = readLogWrittenBy(
        *_mdEngine,
        [&] { thermostatSetup.writeBerendsenInfo(); }
    );

    EXPECT_NE(text.find("Target temperature: 300 K\n"), std::string::npos)
        << text;
    EXPECT_EQ(text.find("K K"), std::string::npos) << text;
}

/**
 * @brief the Langevin log prints the friction in the unit of the input
 * (1/ps), not the internal value in 1/s
 */
TEST_F(TestSetup, langevinLogPrintsTheFrictionInPerPicosecond)
{
    finalizeTargetTemperature(*_settings);
    _settings->thermostatSettings.setFrictionByUser(2.0e12 * Hz);
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    const auto text = readLogWrittenBy(
        *_mdEngine,
        [&] { thermostatSetup.writeLangevinInfo(); }
    );

    EXPECT_NE(text.find("Friction:           2 1/ps"), std::string::npos)
        << text;
    EXPECT_EQ(text.find("K K"), std::string::npos) << text;
}

/**
 * @brief the temperature ramp log prints every temperature with its unit once
 */
TEST_F(TestSetup, temperatureRampLogPrintsTheUnitOnce)
{
    _settings->kinetics.setStartTemperatureByUser(toTemperature(100.0));
    finalizeTargetTemperature(*_settings);
    setup::ThermostatSetup thermostatSetup(*_mdEngine);

    const auto text = readLogWrittenBy(
        *_mdEngine,
        [&] { thermostatSetup.writeTemperatureRampInfo(); }
    );

    EXPECT_NE(text.find("Start temp:          100 K\n"), std::string::npos)
        << text;
    EXPECT_NE(text.find("Target temp:         300 K\n"), std::string::npos)
        << text;
    EXPECT_EQ(text.find("K K"), std::string::npos) << text;
}
