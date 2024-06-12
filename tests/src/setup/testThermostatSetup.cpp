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

#include <gtest/gtest.h>   // for EXPECT_EQ, EXPECT_NO_THROW, InitGo...

#include <cmath>    // for sqrt
#include <string>   // for allocator, basic_string

#include "berendsenThermostat.hpp"           // for BerendsenThermostat
#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _KG_TO_GRAM_
#include "constants/natureConstants.hpp"     // for _UNIVERSAL_GAS_CONSTANT_
#include "exceptions.hpp"             // for InputFileException, customException
#include "gtest/gtest.h"              // for Message, TestPartResult
#include "langevinThermostat.hpp"     // for LangevinThermostat
#include "mdEngine.hpp"               // for MDEngine
#include "noseHooverThermostat.hpp"   // for NoseHooverThermostat
#include "testSetup.hpp"              // for TestSetup
#include "thermostat.hpp"             // for BerendsenThermostat, Thermostat
#include "thermostatSettings.hpp"     // for ThermostatSettings
#include "thermostatSetup.hpp"        // for ThermostatSetup, setupThermostat
#include "throwWithMessage.hpp"       // for EXPECT_THROW_MSG
#include "timingsSettings.hpp"        // for TimingsSettings
#include "velocityRescalingThermostat.hpp"   // for VelocityRescalingThermostat

using namespace setup;

TEST_F(TestSetup, setupThermostat_no_thermostat)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::TimingsSettings::setTimeStep(0.1);
    EXPECT_NO_THROW(thermostatSetup.setup());
}

TEST_F(TestSetup, setupThermostat_both_target_and_end_temp_set)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setEndTemperature(400);
    settings::ThermostatSettings::setTargetTemperature(300);
    settings::ThermostatSettings::setThermostatType("berendsen");
    EXPECT_THROW_MSG(
        thermostatSetup.setup(),
        customException::InputFileException,
        "Both target and end temperature set for berendsen thermostat. They "
        "are mutually exclusive as they are treated as synonyms"
    );

    // needed because otherwise subsequent executions of .setup() will fail
    // because the end temperature is automatically set to the target
    // temperature for consistency
    settings::ThermostatSettings::setEndTemperatureSet(false);
}

TEST_F(TestSetup, setupThermostat_no_target_or_end_temp_set)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setThermostatType("berendsen");
    EXPECT_THROW_MSG(
        thermostatSetup.setup(),
        customException::InputFileException,
        "Target or end temperature not set for berendsen thermostat"
    );
}

TEST_F(TestSetup, setupThermostat_temp_ramping)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::TimingsSettings::setNumberOfSteps(100);

    settings::ThermostatSettings::setThermostatType("berendsen");
    settings::ThermostatSettings::setTargetTemperature(300);
    settings::ThermostatSettings::setStartTemperature(200);
    EXPECT_NO_THROW(thermostatSetup.setup());

    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
        1.0
    );
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
        100
    );

    settings::ThermostatSettings::setTemperatureRampSteps(50);
    settings::ThermostatSettings::setEndTemperatureSet(false);

    EXPECT_NO_THROW(thermostatSetup.setup());

    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
        2.0
    );
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
        50
    );

    settings::ThermostatSettings::setTemperatureRampSteps(0);
    settings::ThermostatSettings::setTemperatureRampFrequency(2);
    settings::ThermostatSettings::setEndTemperatureSet(false);

    EXPECT_NO_THROW(thermostatSetup.setup());

    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
        2.0
    );
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
        100
    );
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getRampingFrequency(),
        2
    );

    settings::ThermostatSettings::setTemperatureRampSteps(200);
    settings::ThermostatSettings::setEndTemperatureSet(false);
    EXPECT_THROW_MSG(
        thermostatSetup.setup(),
        customException::InputFileException,
        "Number of total simulation steps 100 is smaller than the number of "
        "temperature ramping steps 200"
    );

    settings::ThermostatSettings::setTemperatureRampSteps(2);
    settings::ThermostatSettings::setTemperatureRampFrequency(4);
    settings::ThermostatSettings::setEndTemperatureSet(false);
    EXPECT_THROW_MSG(
        thermostatSetup.setup(),
        customException::InputFileException,
        "Temperature ramp frequency 4 is larger than the number of ramping "
        "steps 2"
    );
}

TEST_F(TestSetup, setupThermostat_only_end_temp_defined)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setEndTemperature(300);
    settings::ThermostatSettings::setThermostatType("berendsen");
    EXPECT_NO_THROW(thermostatSetup.setup());
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getTargetTemperature(),
        300
    );
}

TEST_F(TestSetup, setupThermostat_berendsen)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setTargetTemperature(300);
    settings::ThermostatSettings::setTemperatureSet(true);
    EXPECT_NO_THROW(thermostatSetup.setup());
    EXPECT_EQ(
        thermostatSetup.getEngine().getThermostat().getTargetTemperature(),
        300
    );

    const auto berendsenThermostat =
        dynamic_cast<thermostat::BerendsenThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(berendsenThermostat.getTau(), 0.1 * 1000);

    settings::ThermostatSettings::setRelaxationTime(0.2);

    // needed because otherwise subsequent executions of .setup() will fail
    // because the end temperature is automatically set to the target
    // temperature for consistency
    settings::ThermostatSettings::setEndTemperatureSet(false);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto berendsenThermostat2 =
        dynamic_cast<thermostat::BerendsenThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(berendsenThermostat2.getTau(), 0.2 * 1000);

    // needed because otherwise subsequent executions of .setup() will fail
    // because the end temperature is automatically set to the target
    // temperature for consistency
    settings::ThermostatSettings::setEndTemperatureSet(false);
}

TEST_F(TestSetup, setupThermostat_velocity_rescaling)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setThermostatType("velocity_rescaling");
    settings::ThermostatSettings::setTargetTemperature(300);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto velocityRescalingThermostat =
        dynamic_cast<thermostat::VelocityRescalingThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(velocityRescalingThermostat.getTau(), 0.2 * 1000);
}

TEST_F(TestSetup, setupThermostat_langevin)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setThermostatType("langevin");
    settings::ThermostatSettings::setTargetTemperature(300);
    EXPECT_NO_THROW(thermostatSetup.setup());

    // needed because otherwise subsequent executions of .setup() will fail
    // because the end temperature is automatically set to the target
    // temperature for consistency
    settings::ThermostatSettings::setEndTemperatureSet(false);

    const auto langevinThermostat =
        dynamic_cast<thermostat::LangevinThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(langevinThermostat.getFriction(), 1.0e11);

    const auto conversionFactor =
        constants::_UNIVERSAL_GAS_CONSTANT_ *
        constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_ *
        constants::_KG_TO_GRAM_ / constants::_FS_TO_S_;
    const auto sigma = std::sqrt(
        4.0 * langevinThermostat.getFriction() * conversionFactor *
        settings::ThermostatSettings::getTargetTemperature() /
        settings::TimingsSettings::getTimeStep()
    );

    EXPECT_EQ(langevinThermostat.getSigma(), sigma);

    EXPECT_NO_THROW(setupThermostat(*_mdEngine));
}

TEST_F(TestSetup, setupThermostat_nh_chain)
{
    ThermostatSetup thermostatSetup(*_mdEngine);

    settings::ThermostatSettings::setThermostatType("nh-chain");
    settings::ThermostatSettings::setNoseHooverChainLength(5);
    settings::ThermostatSettings::setTargetTemperature(300);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto noseHooverThermostat =
        dynamic_cast<thermostat::NoseHooverThermostat &>(
            thermostatSetup.getEngine().getThermostat()
        );
    EXPECT_EQ(noseHooverThermostat.getCouplingFrequency(), 29979245800000);
    EXPECT_EQ(noseHooverThermostat.getChi().size(), 6);
}
