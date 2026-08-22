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
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "gtest/gtest.h"
#include "hybridInputParser.hpp"
#include "hybridSettings.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace input;
using namespace settings;
using namespace customException;

TEST_F(TestInputFileReader, parseInnerRegionCenter)
{
    auto parser = HybridInputParser{};

    parser.parseInnerRegionCenter({"inner_region_center", "=", "4,2,2"}, 0);

    ASSERT_TRUE(HybridSettings::getInnerRegionCenter().has_value());
    EXPECT_EQ(
        HybridSettings::getInnerRegionCenter(),
        std::optional<std::vector<int>>({2, 4})
    );
}

TEST_F(TestInputFileReader, parseForcedRegionLists)
{
    auto parser = HybridInputParser{};

    parser.parseForcedCoreList({"forced_core_list", "=", "3,1,3"}, 0);
    EXPECT_EQ(HybridSettings::getForcedCoreList(), std::vector<int>({1, 3}));

    parser.parseForcedLayerList({"forced_layer_list", "=", "5,7-9,8"}, 0);
    EXPECT_EQ(
        HybridSettings::getForcedLayerList(),
        std::vector<int>({5, 7, 8, 9})
    );

    parser.parseForcedOuterList({"forced_outer_list", "=", "8-10,9"}, 0);
    EXPECT_EQ(
        HybridSettings::getForcedOuterList(),
        std::vector<int>({8, 9, 10})
    );
}

TEST_F(TestInputFileReader, parseUseQMCharges)
{
    auto parser = HybridInputParser{};

    parser.parseUseQMCharges({"qm_charges", "=", "qm"}, 0);
    EXPECT_TRUE(HybridSettings::getUseQMCharges());

    parser.parseUseQMCharges({"qm_charges", "=", "mm"}, 0);
    EXPECT_FALSE(HybridSettings::getUseQMCharges());

    ASSERT_THROW_MSG(
        parser.parseUseQMCharges({"qm_charges", "=", "invalid"}, 0),
        InputFileException,
        "Invalid qm_charges \"invalid\" in input file\n"
        "Possible values are: qm, mm"
    )
}

TEST_F(TestInputFileReader, parseRegionRadii)
{
    auto parser = HybridInputParser{};

    parser.parseCoreRadius({"core_radius", "=", "3.5"}, 0);
    EXPECT_DOUBLE_EQ(HybridSettings::getCoreRadius(), 3.5);

    parser.parseLayerRadius({"layer_radius", "=", "8.25"}, 0);
    EXPECT_DOUBLE_EQ(HybridSettings::getLayerRadius(), 8.25);

    ASSERT_THROW_MSG(
        parser.parseCoreRadius({"core_radius", "=", "-1.0"}, 0),
        InputFileException,
        "Invalid core_radius -1.0 in input file - must be a positive number"
    )

    ASSERT_THROW_MSG(
        parser.parseLayerRadius({"layer_radius", "=", "-2.0"}, 0),
        InputFileException,
        "Invalid layer_radius -2.0 in input file - must be a positive number"
    )
}

TEST_F(TestInputFileReader, parseThicknesses)
{
    auto parser = HybridInputParser{};

    parser.parseSmoothingRegionThickness(
        {"smoothing_region_thickness", "=", "1.25"},
        0
    );
    EXPECT_DOUBLE_EQ(HybridSettings::getSmoothingRegionThickness(), 1.25);

    parser.parsePointChargeThickness(
        {"point_charge_thickness", "=", "4.75"},
        0
    );
    EXPECT_DOUBLE_EQ(HybridSettings::getPointChargeThickness(), 4.75);

    ASSERT_THROW_MSG(
        parser.parseSmoothingRegionThickness(
            {"smoothing_region_thickness", "=", "-0.1"},
            0
        ),
        InputFileException,
        "Invalid smoothing_region_thickness -0.1 in input file - must be a "
        "positive number"
    )

    ASSERT_THROW_MSG(
        parser.parsePointChargeThickness(
            {"point_charge_thickness", "=", "-0.5"},
            0
        ),
        InputFileException,
        "Invalid point_charge_thickness -0.5 in input file - must be a "
        "positive number"
    )
}

TEST_F(TestInputFileReader, parseSmoothingMethod)
{
    using enum SmoothingMethod;

    auto parser = HybridInputParser{};

    parser.parseSmoothingMethod({"smoothing_method", "=", "hotspot"}, 0);
    EXPECT_EQ(HybridSettings::getSmoothingMethod(), HOTSPOT);

    parser.parseSmoothingMethod({"smoothing_method", "=", "exact"}, 0);
    EXPECT_EQ(HybridSettings::getSmoothingMethod(), EXACT);

    ASSERT_THROW_MSG(
        parser.parseSmoothingMethod({"smoothing_method", "=", "invalid"}, 0),
        InputFileException,
        "Invalid smoothing method \"invalid\" in input file\n"
        "Possible values are: hotspot, exact"
    )
}

TEST_F(TestInputFileReader, parseQMForceDistribution)
{
    using enum QMForceDist;

    auto parser = HybridInputParser{};

    parser.parseQMForceDistribution({"qm_force_distribution", "=", "none"}, 0);
    EXPECT_EQ(HybridSettings::getQMForceDist(), NONE);

    parser.parseQMForceDistribution({"qm_force_distribution", "=", "equal"}, 0);
    EXPECT_EQ(HybridSettings::getQMForceDist(), EQUAL);

    parser.parseQMForceDistribution(
        {"qm_force_distribution", "=", "random"},
        0
    );
    EXPECT_EQ(HybridSettings::getQMForceDist(), RANDOM);

    parser.parseQMForceDistribution(
        {"qm_force_distribution", "=", "distance-weighted"},
        0
    );
    EXPECT_EQ(HybridSettings::getQMForceDist(), DISTANCE_WEIGHTED);

    ASSERT_THROW_MSG(
        parser.parseQMForceDistribution(
            {"qm_force_distribution", "=", "invalid"},
            0
        ),
        InputFileException,
        "Invalid qm force distribution method \"invalid\" in input "
        "file\n"
        "Possible options are: none, equal, random and distance-weighted"
    )
}

TEST_F(TestInputFileReader, parseSelection)
{
    auto parser = HybridInputParser{};

    EXPECT_EQ(
        parser.parseSelection("5,3-4,4,1", "forced_inner_list"),
        std::vector<int>({1, 3, 4, 5})
    );

    EXPECT_EQ(
        parser.parseSelection("", "forced_inner_list"),
        std::vector<int>({0})
    );
}

TEST_F(TestInputFileReader, parseSelectionNoPython)
{
    auto parser = HybridInputParser{};

    EXPECT_EQ(
        parser
            .parseSelectionNoPython(" 7 - 8 , 10, 12 ", "inner_region_center"),
        std::vector<int>({7, 8, 10, 12})
    );

    ASSERT_THROW_MSG(
        parser.parseSelectionNoPython(",", "forced_outer_list"),
        InputFileException,
        "Invalid atom index \"\" for key forced_outer_list. Must be a valid "
        "integer."
    )

    ASSERT_THROW_MSG(
        parser.parseSelectionNoPython("1-a", "forced_outer_list"),
        InputFileException,
        "Invalid end index \"a\" in range \"1-a\" for key "
        "forced_outer_list. Must be a valid integer."
    )
}
