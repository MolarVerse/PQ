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

#include "convergenceSettings.hpp"
#include "gtest/gtest.h"

TEST(ConvSettingsTest, StrategyToString)
{
    using enum settings::ConvStrategy;

    EXPECT_EQ(settings::string(RIGOROUS), "RIGOROUS");
    EXPECT_EQ(settings::string(LOOSE), "LOOSE");
    EXPECT_EQ(settings::string(ABSOLUTE), "ABSOLUTE");
    EXPECT_EQ(settings::string(RELATIVE), "RELATIVE");
    EXPECT_EQ(
        settings::string(static_cast<settings::ConvStrategy>(-1)),
        "none"
    );
}

TEST(ConvSettingsTest, EnergyConvSettersAndOptionalGetters)
{
    settings::ConvSettings::setEnergyConv(1.0e-6);
    ASSERT_TRUE(settings::ConvSettings::getEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getEnergyConv().value(),
        1.0e-6
    );

    settings::ConvSettings::setRelEnergyConv(1.0e-5);
    ASSERT_TRUE(settings::ConvSettings::getRelEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getRelEnergyConv().value(),
        1.0e-5
    );

    settings::ConvSettings::setAbsEnergyConv(1.0e-4);
    ASSERT_TRUE(settings::ConvSettings::getAbsEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getAbsEnergyConv().value(),
        1.0e-4
    );
}

TEST(ConvSettingsTest, ForceConvSettersAndOptionalGetters)
{
    settings::ConvSettings::setForceConv(1.0e-3);
    ASSERT_TRUE(settings::ConvSettings::getForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getForceConv().value(),
        1.0e-3
    );

    settings::ConvSettings::setMaxForceConv(1.0e-2);
    ASSERT_TRUE(settings::ConvSettings::getMaxForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getMaxForceConv().value(),
        1.0e-2
    );

    settings::ConvSettings::setRMSForceConv(1.0e-3);
    ASSERT_TRUE(settings::ConvSettings::getRMSForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvSettings::getRMSForceConv().value(),
        1.0e-3
    );
}
