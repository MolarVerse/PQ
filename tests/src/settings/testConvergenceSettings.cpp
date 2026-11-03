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

TEST(ConvSettingsTest, EnergyConvSettersAndOptionalGetters)
{
    ConvSettings convSettings;
    convSettings.setEnergyConv(1.0e-6);
    ASSERT_TRUE(convSettings.getEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getEnergyConv().value(), 1.0e-6);

    convSettings.setRelEnergyConv(1.0e-5);
    ASSERT_TRUE(convSettings.getRelEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getRelEnergyConv().value(), 1.0e-5);

    convSettings.setAbsEnergyConv(1.0e-4);
    ASSERT_TRUE(convSettings.getAbsEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getAbsEnergyConv().value(), 1.0e-4);
}

TEST(ConvSettingsTest, ForceConvSettersAndOptionalGetters)
{
    ConvSettings convSettings;
    convSettings.setForceConv(1.0e-3);
    ASSERT_TRUE(convSettings.getForceConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getForceConv().value(), 1.0e-3);

    convSettings.setMaxForceConv(1.0e-2);
    ASSERT_TRUE(convSettings.getMaxForceConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getMaxForceConv().value(), 1.0e-2);

    convSettings.setRMSForceConv(1.0e-3);
    ASSERT_TRUE(convSettings.getRMSForceConv().has_value());
    EXPECT_DOUBLE_EQ(convSettings.getRMSForceConv().value(), 1.0e-3);
}
