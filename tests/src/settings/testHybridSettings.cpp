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
    settings::HybridSettings::setInnerRegionCenter({4, 2, 9});

    ASSERT_TRUE(settings::HybridSettings::getInnerRegionCenter().has_value());
    EXPECT_EQ(
        settings::HybridSettings::getInnerRegionCenter(),
        std::optional<std::vector<size_t>>({4, 2, 9})
    );
}

TEST(HybridSettingsTest, ForcedRegionListsRoundTrip)
{
    settings::HybridSettings::setForcedCoreList({1, 3, 5});
    EXPECT_EQ(
        settings::HybridSettings::getForcedCoreList(),
        std::vector<int>({1, 3, 5})
    );

    settings::HybridSettings::setForcedLayerList({7, 9, 11});
    EXPECT_EQ(
        settings::HybridSettings::getForcedLayerList(),
        std::vector<int>({7, 9, 11})
    );

    settings::HybridSettings::setForcedOuterList({2, 4, 6});
    EXPECT_EQ(
        settings::HybridSettings::getForcedOuterList(),
        std::vector<int>({2, 4, 6})
    );
}

TEST(HybridSettingsTest, BoolAndRadiusSettingsRoundTrip)
{
    settings::HybridSettings::setUseQMCharges(false);
    EXPECT_FALSE(settings::HybridSettings::getUseQMCharges());

    settings::HybridSettings::setUseQMCharges(true);
    EXPECT_TRUE(settings::HybridSettings::getUseQMCharges());

    settings::HybridSettings::setCoreRadius(2.5);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getCoreRadius(), 2.5);

    settings::HybridSettings::setLayerRadius(6.75);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getLayerRadius(), 6.75);

    settings::HybridSettings::setSmoothingRegionThickness(1.25);
    EXPECT_DOUBLE_EQ(
        settings::HybridSettings::getSmoothingRegionThickness(),
        1.25
    );

    settings::HybridSettings::setPointChargeThickness(4.5);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getPointChargeThickness(), 4.5);
}

TEST(HybridSettingsTest, EnumSettingsRoundTrip)
{
    using enum SmoothingMethod;
    using enum QMForceDist;

    settings::HybridSettings::setSmoothingMethod(HOTSPOT);
    EXPECT_EQ(settings::HybridSettings::getSmoothingMethod(), HOTSPOT);

    settings::HybridSettings::setSmoothingMethod(EXACT);
    EXPECT_EQ(settings::HybridSettings::getSmoothingMethod(), EXACT);

    settings::HybridSettings::setQMForceDist(NONE);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), NONE);

    settings::HybridSettings::setQMForceDist(EQUAL);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), EQUAL);

    settings::HybridSettings::setQMForceDist(RANDOM);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), RANDOM);

    settings::HybridSettings::setQMForceDist(DISTANCE_WEIGHTED);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), DISTANCE_WEIGHTED);
}
