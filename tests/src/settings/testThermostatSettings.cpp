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

// for Message, TestPartResult
#include "thermostatSettings.hpp"

TEST(ThermostatSettingsTest, SetThermostatTypeTest)
{
    settings::ThermostatSettings::setThermostatType("berendsen");
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::BERENDSEN
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(
            settings::ThermostatSettings::getThermostatType()
        ),
        "BERENDSEN"
    );

    settings::ThermostatSettings::setThermostatType("none");
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::NONE
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(
            settings::ThermostatSettings::getThermostatType()
        ),
        "NONE"
    );

    settings::ThermostatSettings::setThermostatType("langevin");
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::LANGEVIN
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(
            settings::ThermostatSettings::getThermostatType()
        ),
        "LANGEVIN"
    );

    settings::ThermostatSettings::setThermostatType("NH-chain");
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::NOSE_HOOVER
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(
            settings::ThermostatSettings::getThermostatType()
        ),
        "NOSE_HOOVER"
    );

    settings::ThermostatSettings::setThermostatType("velocity_rescaling");
    EXPECT_EQ(
        settings::ThermostatSettings::getThermostatType(),
        ThermostatType::VELOCITY_RESCALING
    );
    EXPECT_EQ(
        ThermostatTypeMeta::toString(
            settings::ThermostatSettings::getThermostatType()
        ),
        "VELOCITY_RESCALING"
    );
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
