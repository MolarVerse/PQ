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
#include "units.hpp"

TEST(ConvSettingsTest, EnergyConvSettersAndOptionalGetters)
{
    ConvSettings convSettings;
    convSettings.setRelEnergyConvByUser(1.0e-5);
    EXPECT_DOUBLE_EQ(convSettings.getRelEnergyConv(), 1.0e-5);

    convSettings.setAbsEnergyConvByUser(1.0e-4 * kcal_per_mol);
    EXPECT_DOUBLE_EQ(
        convSettings.getAbsEnergyConv().numerical_value_in(kcal_per_mol),
        1.0e-4
    );
}

TEST(ConvSettingsTest, ForceConvSettersAndOptionalGetters)
{
    ConvSettings convSettings;
    convSettings.setMaxForceConvByUser(1.0e-2 * kcal_per_mol / angstrom);
    EXPECT_DOUBLE_EQ(
        convSettings.getMaxForceConv().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.0e-2
    );

    convSettings.setRMSForceConvByUser(1.0e-3 * kcal_per_mol / angstrom);
    EXPECT_DOUBLE_EQ(
        convSettings.getRMSForceConv().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.0e-3
    );
}
