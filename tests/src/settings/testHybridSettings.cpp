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

#include <optional>
#include <vector>

#include "hybridSettings.hpp"

TEST(HybridSettingsTest, InnerRegionCenterRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setInnerRegionCenter({4, 2, 9});

    ASSERT_TRUE(hybridSettings.getInnerRegionCenter().has_value());
    EXPECT_EQ(
        hybridSettings.getInnerRegionCenter(),
        std::optional<std::vector<size_t>>({4, 2, 9})
    );
}

TEST(HybridSettingsTest, ForcedRegionListsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setForcedCoreList({1, 3, 5});
    EXPECT_EQ(hybridSettings.getForcedCoreList(), std::vector<int>({1, 3, 5}));

    hybridSettings.setForcedLayerList({7, 9, 11});
    EXPECT_EQ(
        hybridSettings.getForcedLayerList(),
        std::vector<int>({7, 9, 11})
    );

    hybridSettings.setForcedOuterList({2, 4, 6});
    EXPECT_EQ(hybridSettings.getForcedOuterList(), std::vector<int>({2, 4, 6}));
}

TEST(HybridSettingsTest, BoolAndRadiusSettingsRoundTrip)
{
    HybridSettings hybridSettings;
    hybridSettings.setUseQMCharges(false);
    EXPECT_FALSE(hybridSettings.getUseQMCharges());

    hybridSettings.setUseQMCharges(true);
    EXPECT_TRUE(hybridSettings.getUseQMCharges());

    hybridSettings.setCoreRadius(2.5);
    EXPECT_DOUBLE_EQ(hybridSettings.getCoreRadius(), 2.5);

    hybridSettings.setLayerRadius(6.75);
    EXPECT_DOUBLE_EQ(hybridSettings.getLayerRadius(), 6.75);

    hybridSettings.setSmoothingRegionThickness(1.25);
    EXPECT_DOUBLE_EQ(hybridSettings.getSmoothingRegionThickness(), 1.25);

    hybridSettings.setPointChargeThickness(4.5);
    EXPECT_DOUBLE_EQ(hybridSettings.getPointChargeThickness(), 4.5);
}

TEST(HybridSettingsTest, EnumSettingsRoundTrip)
{
    using enum SmoothingMethod;
    using enum QMForceDist;

    HybridSettings hybridSettings;
    hybridSettings.setSmoothingMethod(HOTSPOT);
    EXPECT_EQ(hybridSettings.getSmoothingMethod(), HOTSPOT);

    hybridSettings.setSmoothingMethod(EXACT);
    EXPECT_EQ(hybridSettings.getSmoothingMethod(), EXACT);

    hybridSettings.setQMForceDist(NONE);
    EXPECT_EQ(hybridSettings.getQMForceDist(), NONE);

    hybridSettings.setQMForceDist(EQUAL);
    EXPECT_EQ(hybridSettings.getQMForceDist(), EQUAL);

    hybridSettings.setQMForceDist(RANDOM);
    EXPECT_EQ(hybridSettings.getQMForceDist(), RANDOM);

    hybridSettings.setQMForceDist(DISTANCE_WEIGHTED);
    EXPECT_EQ(hybridSettings.getQMForceDist(), DISTANCE_WEIGHTED);
}
