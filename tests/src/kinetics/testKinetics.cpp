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

// #include "kinetics.hpp"
// #include "thermostatSettings.hpp"
// #include "unit_types.hpp"
// #include "units.hpp"

// class TestKinetics : public ::testing::Test
// {
//    protected:
//     KineticsSettings _settings;
// };

// TEST_F(TestKinetics, applyTemperatureRamping)
// {
//     _settings.setTargetTemperature(mstd::units::point<K>(300.0));

//     kinetics::Kinetics kinetics{_settings};
//     kinetics.applyTemperatureRamping();
//     EXPECT_EQ(kinetics.getTargetTemperature(), mstd::units::point<K>(300.0));

//     _settings.setRampingStepsByUser(1);
//     _settings.setRampingFrequencyByUser(1);

//     Kinetics kinetics{_settings};
//     kinetics.applyTemperatureRamping();
//     EXPECT_EQ(kinetics.getTargetTemperature(), mstd::units::point<K>(301.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(301.0));

//     _kinetics.setTemperatureIncrease(mstd::units::delta<K>(1.0));
//     _settings.setRampingStepsByUser(2);
//     _settings.setRampingFrequencyByUser(1);
//     _kinetics.setTargetTemperature(mstd::units::point<K>(300.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(301.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(302.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(302.0));

//     _kinetics.setTemperatureIncrease(mstd::units::delta<K>(1.0));
//     _settings.setRampingStepsByUser(4);
//     _settings.setRampingFrequencyByUser(2);
//     _kinetics.setTargetTemperature(mstd::units::point<K>(300.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(300.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(301.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(301.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(302.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(302.0));

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(_kinetics.getTargetTemperature(),
//     mstd::units::point<K>(302.0));
// }

// TEST_F(TestKinetics, temperatureRampingUpdatesActualTargetTemperature)
// {
//     ThermostatSettings::setActualTargetTemperature(toTemperature(300.0));

//     _kinetics.setTargetTemperature(mstd::units::point<K>(300.0));
//     _kinetics.setTemperatureIncrease(mstd::units::delta<K>(2.5));
//     _settings.setRampingStepsByUser(2);
//     _settings.setRampingFrequencyByUser(1);

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(
//         ThermostatSettings::getActualTargetTemperature(),
//         toTemperature(302.5)
//     );

//     _kinetics.applyTemperatureRamping();
//     EXPECT_EQ(
//         ThermostatSettings::getActualTargetTemperature(),
//         toTemperature(305.0)
//     );

//     ThermostatSettings::setActualTargetTemperature(toTemperature(0.0));
// }
