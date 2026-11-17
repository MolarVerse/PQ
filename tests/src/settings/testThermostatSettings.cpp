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

#include "enums/thermostat.hpp"
#include "thermostatSettings.hpp"

TEST(ThermostatSettingsTest, SetThermostatTypeTest)
{
    ThermostatSettings thermostatSettings;
    thermostatSettings.setThermostatTypeByUser(ThermostatType::BERENDSEN);
    EXPECT_EQ(
        thermostatSettings.getThermostatType(),
        ThermostatType::BERENDSEN
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(thermostatSettings.getThermostatType()),
        "BERENDSEN"
    );

    thermostatSettings.setThermostatTypeByUser(ThermostatType::NONE);
    EXPECT_EQ(thermostatSettings.getThermostatType(), ThermostatType::NONE);
    EXPECT_EQ(
        ThermostatTypeMeta::toString(thermostatSettings.getThermostatType()),
        "NONE"
    );

    thermostatSettings.setThermostatTypeByUser(ThermostatType::LANGEVIN);
    EXPECT_EQ(thermostatSettings.getThermostatType(), ThermostatType::LANGEVIN);
    EXPECT_EQ(
        ThermostatTypeMeta::toString(thermostatSettings.getThermostatType()),
        "LANGEVIN"
    );

    thermostatSettings.setThermostatTypeByUser(ThermostatType::NOSE_HOOVER);
    EXPECT_EQ(
        thermostatSettings.getThermostatType(),
        ThermostatType::NOSE_HOOVER
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(thermostatSettings.getThermostatType()),
        "NOSE_HOOVER"
    );

    thermostatSettings.setThermostatTypeByUser(
        ThermostatType::VELOCITY_RESCALING
    );
    EXPECT_EQ(
        thermostatSettings.getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(thermostatSettings.getThermostatType()),
        "VELOCITY_RESCALING"
    );
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
