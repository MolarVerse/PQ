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

TEST(HybridSettingsTest, InnerRegionCenterRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setInnerRegionCenterByUser({4, 2, 9});

    EXPECT_EQ(
        hybridSettings.getInnerRegionCenter(),
        std::vector<size_t>({4, 2, 9})
    );
}

TEST(HybridSettingsTest, ForcedRegionListsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setForcedCoreListByUser({1, 3, 5});
    EXPECT_EQ(hybridSettings.getForcedCoreList(), std::vector<int>({1, 3, 5}));

    hybridSettings.setForcedLayerListByUser({7, 9, 11});
    EXPECT_EQ(
        hybridSettings.getForcedLayerList(),
        std::vector<int>({7, 9, 11})
    );

    hybridSettings.setForcedOuterListByUser({2, 4, 6});
    EXPECT_EQ(hybridSettings.getForcedOuterList(), std::vector<int>({2, 4, 6}));
}

TEST(HybridSettingsTest, BoolAndRadiusSettingsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setUseQMChargesByUser(false);
    EXPECT_FALSE(hybridSettings.getUseQMCharges());

    hybridSettings.setUseQMChargesByUser(true);
    EXPECT_TRUE(hybridSettings.getUseQMCharges());

    hybridSettings.setCoreRadiusByUser(2.5);
    EXPECT_DOUBLE_EQ(hybridSettings.getCoreRadius(), 2.5);

    hybridSettings.setLayerRadiusByUser(6.75);
    EXPECT_DOUBLE_EQ(hybridSettings.getLayerRadius(), 6.75);

    hybridSettings.setSmoothingRegionThicknessByUser(1.25);
    EXPECT_DOUBLE_EQ(hybridSettings.getSmoothingRegionThickness(), 1.25);

    hybridSettings.setPointChargeThicknessByUser(4.5);
    EXPECT_DOUBLE_EQ(hybridSettings.getPointChargeThickness(), 4.5);
}

TEST(HybridSettingsTest, EnumSettingsRoundTrip)
{
    using enum SmoothingMethod;
    using enum QMForceDist;

    HybridSettings hybridSettings;
    hybridSettings.setSmoothingMethodByUser(HOTSPOT);
    EXPECT_EQ(hybridSettings.getSmoothingMethod(), HOTSPOT);

    hybridSettings.setSmoothingMethodByUser(EXACT);
    EXPECT_EQ(hybridSettings.getSmoothingMethod(), EXACT);

    hybridSettings.setQMForceDistByUser(EQUAL);
    EXPECT_EQ(hybridSettings.getQMForceDist(), EQUAL);

    hybridSettings.setQMForceDistByUser(RANDOM);
    EXPECT_EQ(hybridSettings.getQMForceDist(), RANDOM);

    hybridSettings.setQMForceDistByUser(DISTANCE_WEIGHTED);
    EXPECT_EQ(hybridSettings.getQMForceDist(), DISTANCE_WEIGHTED);
}
