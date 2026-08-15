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

using namespace settings;

TEST(HybridSettingsTest, StringRoundTripForSmoothingMethod)
{
    using enum SmoothingMethod;

    EXPECT_EQ(string(HOTSPOT), "Hotspot");
    EXPECT_EQ(string(EXACT), "Exact");
    EXPECT_EQ(string(static_cast<SmoothingMethod>(-1)), "NONE");
}

TEST(HybridSettingsTest, InnerRegionCenterRoundTrip)
{
    HybridSettings::setInnerRegionCenter({4, 2, 9});

    ASSERT_TRUE(HybridSettings::getInnerRegionCenter().has_value());
    EXPECT_EQ(
        HybridSettings::getInnerRegionCenter(),
        std::optional<std::vector<int>>({4, 2, 9})
    );
}

TEST(HybridSettingsTest, ForcedRegionListsRoundTrip)
{
    HybridSettings::setForcedCoreList({1, 3, 5});
    EXPECT_EQ(HybridSettings::getForcedCoreList(), std::vector<int>({1, 3, 5}));

    HybridSettings::setForcedLayerList({7, 9, 11});
    EXPECT_EQ(
        HybridSettings::getForcedLayerList(),
        std::vector<int>({7, 9, 11})
    );

    HybridSettings::setForcedOuterList({2, 4, 6});
    EXPECT_EQ(
        HybridSettings::getForcedOuterList(),
        std::vector<int>({2, 4, 6})
    );
}

TEST(HybridSettingsTest, BoolAndRadiusSettingsRoundTrip)
{
    HybridSettings::setUseQMCharges(false);
    EXPECT_FALSE(HybridSettings::getUseQMCharges());

    HybridSettings::setUseQMCharges(true);
    EXPECT_TRUE(HybridSettings::getUseQMCharges());

    HybridSettings::setCoreRadius(2.5);
    EXPECT_DOUBLE_EQ(HybridSettings::getCoreRadius(), 2.5);

    HybridSettings::setLayerRadius(6.75);
    EXPECT_DOUBLE_EQ(HybridSettings::getLayerRadius(), 6.75);

    HybridSettings::setSmoothingRegionThickness(1.25);
    EXPECT_DOUBLE_EQ(HybridSettings::getSmoothingRegionThickness(), 1.25);

    HybridSettings::setPointChargeThickness(4.5);
    EXPECT_DOUBLE_EQ(HybridSettings::getPointChargeThickness(), 4.5);
}

TEST(HybridSettingsTest, EnumSettingsRoundTrip)
{
    using enum SmoothingMethod;
    using enum QMForceDist;

    HybridSettings::setSmoothingMethod(HOTSPOT);
    EXPECT_EQ(HybridSettings::getSmoothingMethod(), HOTSPOT);

    HybridSettings::setSmoothingMethod(EXACT);
    EXPECT_EQ(HybridSettings::getSmoothingMethod(), EXACT);

    HybridSettings::setQMForceDist(NONE);
    EXPECT_EQ(HybridSettings::getQMForceDist(), NONE);

    HybridSettings::setQMForceDist(EQUAL);
    EXPECT_EQ(HybridSettings::getQMForceDist(), EQUAL);

    HybridSettings::setQMForceDist(RANDOM);
    EXPECT_EQ(HybridSettings::getQMForceDist(), RANDOM);

    HybridSettings::setQMForceDist(DISTANCE_WEIGHTED);
    EXPECT_EQ(HybridSettings::getQMForceDist(), DISTANCE_WEIGHTED);
}
