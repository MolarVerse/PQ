// /*****************************************************************************
// <GPL_HEADER>

//     PQ
//     Copyright (C) 2023-now  Jakob Gamper

//     This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.

//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.

//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.

// <GPL_HEADER>
// ******************************************************************************/

// #include <gtest/gtest.h>

// #include <cmath>

// #include "berendsenThermostat.hpp"
// #include "constants/conversionFactors.hpp"
// #include "constants/natureConstants.hpp"
// #include "exceptions.hpp"
// #include "inputFileReader.hpp"
// #include "langevinThermostat.hpp"
// #include "noseHooverThermostat.hpp"
// #include "testSetup.hpp"
// #include "thermostatSettings.hpp"
// #include "thermostatSetup.hpp"
// #include "throwWithMessage.hpp"
// #include "timingsSettings.hpp"
// #include "units.hpp"
// #include "velocityRescalingThermostat.hpp"

// TEST_F(TestSetup, setupThermostatNoThermostat)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     settings::TimingsSettings::setTimeStep(0.1 * fs);
//     EXPECT_NO_THROW(thermostatSetup.setup());
// }

// TEST_F(TestSetup, setupThermostatTempRamping)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     settings::TimingsSettings::setNumberOfSteps(100);

//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::BERENDSEN
//     );
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     ThermostatSettings::setStartTemperature(toTemperature(200));
//     EXPECT_NO_THROW(thermostatSetup.setup());

//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
//         mstd::units::delta<K>(1.0)
//     );
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
//         100
//     );

//     ThermostatSettings::setTemperatureRampSteps(50);
//     ThermostatSettings::setEndTemperatureSet(false);

//     EXPECT_NO_THROW(thermostatSetup.setup());

//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
//         mstd::units::delta<K>(2.0)
//     );
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
//         50
//     );

//     ThermostatSettings::setTemperatureRampSteps(0);
//     ThermostatSettings::setTemperatureRampFrequency(2);
//     ThermostatSettings::setEndTemperatureSet(false);

//     EXPECT_NO_THROW(thermostatSetup.setup());

//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
//         mstd::units::delta<K>(2.0)
//     );
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getRampingStepsLeft(),
//         100
//     );
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getRampingFrequency(),
//         2
//     );
// }

// TEST_F(TestSetup, temperatureRampReachesEndWithPartialFinalInterval)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);
//     input::InputFileReader reader("input.in", *_settings);

//     reader.process({"nstep", "=", "10"});
//     reader.process({"thermostat", "=", "berendsen"});
//     reader.process({"temp", "=", "300"});
//     reader.process({"start_temp", "=", "200"});
//     reader.process({"temp_ramp_steps", "=", "10"});
//     reader.process({"temp_ramp_frequency", "=", "3"});

//     thermostatSetup.setup();

//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTemperatureIncrease(),
//         mstd::units::delta<K>(25.0)
//     );

//     for (size_t step = 0; step < 10; ++step)
//         thermostatSetup.getEngine().getThermostat().applyTemperatureRamping();

//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTargetTemperature(),
//         mstd::units::point<K>(300.0)
//     );

//     ThermostatSettings::setTemperatureRampSteps(0);
//     ThermostatSettings::setTemperatureRampFrequency(1);
// }

// TEST_F(TestSetup, rejectsEmptyTemperatureRamp)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     settings::TimingsSettings::setNumberOfSteps(0);
//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::BERENDSEN
//     );
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     ThermostatSettings::setStartTemperature(toTemperature(200));
//     ThermostatSettings::setTemperatureRampSteps(0);

//     EXPECT_THROW_MSG(
//         thermostatSetup.setup(),
//         exc::InputFileException,
//         "Temperature ramp requires at least one simulation step"
//     );
// }

// TEST_F(TestSetup, rejectsZeroTemperatureRampFrequency)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     settings::TimingsSettings::setNumberOfSteps(10);
//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::BERENDSEN
//     );
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     ThermostatSettings::setStartTemperature(toTemperature(200));
//     ThermostatSettings::setTemperatureRampSteps(10);
//     ThermostatSettings::setTemperatureRampFrequency(0);

//     EXPECT_THROW_MSG(
//         thermostatSetup.setup(),
//         exc::InputFileException,
//         "Temperature ramp frequency must be greater than zero"
//     );

//     ThermostatSettings::setTemperatureRampSteps(0);
//     ThermostatSettings::setTemperatureRampFrequency(1);
// }

// TEST_F(TestSetup, setupThermostatOnlyEndTempDefined)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     ThermostatSettings::setEndTemperature(toTemperature(300));
//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::BERENDSEN
//     );
//     EXPECT_NO_THROW(thermostatSetup.setup());
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTargetTemperature(),
//         toTemperature(300)
//     );
// }

// TEST_F(TestSetup, setupThermostatBerendsen)
// {
//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::BERENDSEN
//     );
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     ThermostatSettings::setTemperatureSet(true);
//     EXPECT_NO_THROW(thermostatSetup.setup());
//     EXPECT_EQ(
//         thermostatSetup.getEngine().getThermostat().getTargetTemperature(),
//         toTemperature(300)
//     );

//     const auto berendsenThermostat =
//         dynamic_cast<thermostat::BerendsenThermostat &>(
//             thermostatSetup.getEngine().getThermostat()
//         );
//     EXPECT_EQ(berendsenThermostat.getTau(), 0.1 * 1000 * fs);

//     ThermostatSettings::setRelaxationTime(0.2 * ps);

//     // needed because otherwise subsequent executions of .setup() will fail
//     // because the end temperature is automatically set to the target
//     // temperature for consistency
//     ThermostatSettings::setEndTemperatureSet(false);
//     EXPECT_NO_THROW(thermostatSetup.setup());

//     const auto berendsenThermostat2 =
//         dynamic_cast<thermostat::BerendsenThermostat &>(
//             thermostatSetup.getEngine().getThermostat()
//         );
//     EXPECT_EQ(berendsenThermostat2.getTau(), 0.2 * 1000 * fs);

//     // needed because otherwise subsequent executions of .setup() will fail
//     // because the end temperature is automatically set to the target
//     // temperature for consistency
//     ThermostatSettings::setEndTemperatureSet(false);
// }

// TEST_F(TestSetup, setupThermostatVelocityRescaling)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::VELOCITY_RESCALING
//     );
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     ThermostatSettings::setRelaxationTime(0.2 * ps);
//     EXPECT_NO_THROW(thermostatSetup.setup());

//     const auto velocityRescalingThermostat =
//         dynamic_cast<thermostat::VelocityRescalingThermostat &>(
//             thermostatSetup.getEngine().getThermostat()
//         );
//     EXPECT_EQ(velocityRescalingThermostat.getTau(), 0.2 * 1000 * fs);
// }

// TEST_F(TestSetup, setupThermostatLangevin)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::LANGEVIN
//     );
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     EXPECT_NO_THROW(thermostatSetup.setup());

//     // needed because otherwise subsequent executions of .setup() will fail
//     // because the end temperature is automatically set to the target
//     // temperature for consistency
//     ThermostatSettings::setEndTemperatureSet(false);

//     const auto langevinThermostat =
//         dynamic_cast<thermostat::LangevinThermostat &>(
//             thermostatSetup.getEngine().getThermostat()
//         );
//     EXPECT_EQ(langevinThermostat.getFriction(), 1.0e11);

//     const auto conversionFactor =
//         UNIVERSAL_GAS_CONSTANT * M2_TO_ANGSTROM2 * KG_TO_GRAM / FS_TO_S;
//     const auto sigma = std::sqrt(
//         4.0 * langevinThermostat.getFriction() * conversionFactor *
//         ThermostatSettings::getTargetTemperature()
//             .quantity_from_zero()
//             .numerical_value_in(K) /
//         settings::TimingsSettings::getTimeStep().numerical_value_in(fs)
//     );

//     EXPECT_EQ(langevinThermostat.getSigma(), sigma);

//     EXPECT_NO_THROW(setup::setupThermostat(*_mdEngine));
// }

// TEST_F(TestSetup, setupThermostatNhChain)
// {
//     setup::ThermostatSetup thermostatSetup(*_mdEngine);

//     _settings->thermostatSettings.setThermostatTypeByUser(
//         ThermostatType::NOSE_HOOVER
//     );
//     _settings->noseHooverSettings.setNoseHooverChainLengthByUser(5);
//     ThermostatSettings::setTargetTemperature(toTemperature(300));
//     EXPECT_NO_THROW(thermostatSetup.setup());

//     const auto noseHooverThermostat =
//         dynamic_cast<thermostat::NoseHooverThermostat &>(
//             thermostatSetup.getEngine().getThermostat()
//         );
//     EXPECT_EQ(noseHooverThermostat.getChi().size(), 5);
//     EXPECT_EQ(noseHooverThermostat.getZeta().size(), 5);
// }
