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

#include <vector>

#include "hybridSettings.hpp"
#include "units.hpp"

TEST(HybridSettingsTest, InnerRegionCenterRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.set_innerRegionCenterByUser({4, 2, 9});

    EXPECT_EQ(
        hybridSettings.get_innerRegionCenter(),
        std::vector<size_t>({4, 2, 9})
    );
}

TEST(HybridSettingsTest, ForcedRegionListsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.set_forcedCoreListByUser({1, 3, 5});
    EXPECT_EQ(hybridSettings.get_forcedCoreList(), std::vector<int>({1, 3, 5}));

    hybridSettings.set_forcedLayerListByUser({7, 9, 11});
    EXPECT_EQ(
        hybridSettings.get_forcedLayerList(),
        std::vector<int>({7, 9, 11})
    );

    hybridSettings.set_forcedOuterListByUser({2, 4, 6});
    EXPECT_EQ(
        hybridSettings.get_forcedOuterList(),
        std::vector<int>({2, 4, 6})
    );
}

TEST(HybridSettingsTest, BoolAndRadiusSettingsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.set_useQMChargesByUser(false);
    EXPECT_FALSE(hybridSettings.get_useQMCharges());

    hybridSettings.set_useQMChargesByUser(true);
    EXPECT_TRUE(hybridSettings.get_useQMCharges());

    hybridSettings.set_coreRadiusByUser(2.5 * angstrom);
    EXPECT_DOUBLE_EQ(
        hybridSettings.get_coreRadius().numerical_value_in(angstrom),
        2.5
    );

    hybridSettings.set_layerRadiusByUser(6.75 * angstrom);
    EXPECT_DOUBLE_EQ(
        hybridSettings.get_layerRadius().numerical_value_in(angstrom),
        6.75
    );

    hybridSettings.set_smoothingRegionThicknessByUser(1.25 * angstrom);
    EXPECT_DOUBLE_EQ(
        hybridSettings.get_smoothingRegionThickness().numerical_value_in(
            angstrom
        ),
        1.25
    );

    hybridSettings.set_pointChargeThicknessByUser(4.5 * angstrom);
    EXPECT_DOUBLE_EQ(
        hybridSettings.get_pointChargeThickness().numerical_value_in(angstrom),
        4.5
    );
}

TEST(HybridSettingsTest, EnumSettingsRoundTrip)
{
    using enum SmoothingMethod;
    using enum QMForceDist;

    HybridSettings hybridSettings;
    hybridSettings.set_smoothingByUser(HOTSPOT);
    EXPECT_EQ(hybridSettings.get_smoothing(), HOTSPOT);

    hybridSettings.set_smoothingByUser(EXACT);
    EXPECT_EQ(hybridSettings.get_smoothing(), EXACT);

    hybridSettings.set_qmForceDistByUser(EQUAL);
    EXPECT_EQ(hybridSettings.get_qmForceDist(), EQUAL);

    hybridSettings.set_qmForceDistByUser(RANDOM);
    EXPECT_EQ(hybridSettings.get_qmForceDist(), RANDOM);

    hybridSettings.set_qmForceDistByUser(DISTANCE_WEIGHTED);
    EXPECT_EQ(hybridSettings.get_qmForceDist(), DISTANCE_WEIGHTED);
}
