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

TEST(ConvergenceSettingsTest, EnergyConvSettersAndOptionalGetters)
{
    settings::ConvergenceSettings::setEnergyConv(1.0e-6);
    ASSERT_TRUE(settings::ConvergenceSettings::getEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getEnergyConv().value(),
        1.0e-6
    );

    settings::ConvergenceSettings::setRelEnergyConv(1.0e-5);
    ASSERT_TRUE(settings::ConvergenceSettings::getRelEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getRelEnergyConv().value(),
        1.0e-5
    );

    settings::ConvergenceSettings::setAbsEnergyConv(1.0e-4);
    ASSERT_TRUE(settings::ConvergenceSettings::getAbsEnergyConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getAbsEnergyConv().value(),
        1.0e-4
    );
}

TEST(ConvergenceSettingsTest, ForceConvSettersAndOptionalGetters)
{
    settings::ConvergenceSettings::setForceConv(1.0e-3);
    ASSERT_TRUE(settings::ConvergenceSettings::getForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getForceConv().value(),
        1.0e-3
    );

    settings::ConvergenceSettings::setMaxForceConv(1.0e-2);
    ASSERT_TRUE(settings::ConvergenceSettings::getMaxForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getMaxForceConv().value(),
        1.0e-2
    );

    settings::ConvergenceSettings::setRMSForceConv(1.0e-3);
    ASSERT_TRUE(settings::ConvergenceSettings::getRMSForceConv().has_value());
    EXPECT_DOUBLE_EQ(
        settings::ConvergenceSettings::getRMSForceConv().value(),
        1.0e-3
    );
}
