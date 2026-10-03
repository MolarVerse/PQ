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
#include "hybridInputParser.hpp"
#include "hybridSettings.hpp"
#include "inputConverter.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

TEST_F(TestInputFileReader, parseInnerRegionCenter)
{
    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("inner_region_center"));
    const auto& parseFunc = funcMap.at("inner_region_center");

    parseFunc({"inner_region_center", "=", "4,2,2"}, 0);
    ASSERT_TRUE(settings::HybridSettings::getInnerRegionCenter().has_value());
    EXPECT_EQ(
        settings::HybridSettings::getInnerRegionCenter(),
        std::optional<std::vector<size_t>>({2, 4})
    );
}

TEST_F(TestInputFileReader, parseForcedRegionLists)
{
    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("forced_core_list"));
    ASSERT_TRUE(funcMap.contains("forced_layer_list"));
    ASSERT_TRUE(funcMap.contains("forced_outer_list"));
    const auto& parseForcedCoreListFunc  = funcMap.at("forced_core_list");
    const auto& parseForcedLayerListFunc = funcMap.at("forced_layer_list");
    const auto& parseForcedOuterListFunc = funcMap.at("forced_outer_list");

    parseForcedCoreListFunc({"forced_core_list", "=", "3,1,3"}, 0);
    EXPECT_EQ(
        settings::HybridSettings::getForcedCoreList(),
        std::vector<int>({1, 3})
    );

    _clearParser(parser);

    parseForcedLayerListFunc({"forced_layer_list", "=", "5,7-9,8"}, 0);
    EXPECT_EQ(
        settings::HybridSettings::getForcedLayerList(),
        std::vector<int>({5, 7, 8, 9})
    );

    _clearParser(parser);

    parseForcedOuterListFunc({"forced_outer_list", "=", "8-10,9"}, 0);
    EXPECT_EQ(
        settings::HybridSettings::getForcedOuterList(),
        std::vector<int>({8, 9, 10})
    );
}

TEST_F(TestInputFileReader, parseUseQMCharges)
{
    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("use_qm_charges"));
    const auto& parseUseQMChargesFunc = funcMap.at("use_qm_charges");

    parseUseQMChargesFunc({"qm_charges", "=", "qm"}, 0);
    EXPECT_TRUE(settings::HybridSettings::getUseQMCharges());

    _clearParser(parser);

    parseUseQMChargesFunc({"qm_charges", "=", "mm"}, 0);
    EXPECT_FALSE(settings::HybridSettings::getUseQMCharges());

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseUseQMChargesFunc({"qm_charges", "=", "invalid"}, 0),
        exc::InputFileException,
        "Invalid value \"invalid\" for key \"use_qm_charges\" at line 0 in "
        "input file. Allowed values: qm, mm"
    )
}

TEST_F(TestInputFileReader, parseRegionRadii)
{
    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("core_radius"));
    ASSERT_TRUE(funcMap.contains("layer_radius"));
    const auto& parseCoreRadiusFunc  = funcMap.at("core_radius");
    const auto& parseLayerRadiusFunc = funcMap.at("layer_radius");

    parseCoreRadiusFunc({"core_radius", "=", "3.5"}, 0);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getCoreRadius(), 3.5);

    _clearParser(parser);

    parseLayerRadiusFunc({"layer_radius", "=", "8.25"}, 0);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getLayerRadius(), 8.25);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseCoreRadiusFunc({"core_radius", "=", "-1.0"}, 0),
        exc::InputFileException,
        "Invalid value \"-1.0\" for key \"core_radius\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 0"
    )

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseLayerRadiusFunc({"layer_radius", "=", "-2.0"}, 0),
        exc::InputFileException,
        "Invalid value \"-2.0\" for key \"layer_radius\" at line 0 in input "
        "file: failed validation with message Value must be greater than or "
        "equal to 0"
    )
}

TEST_F(TestInputFileReader, parseThicknesses)
{
    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("smoothing_region_thickness"));
    ASSERT_TRUE(funcMap.contains("point_charge_thickness"));
    const auto& parseSmoothingRegionThicknessFunc =
        funcMap.at("smoothing_region_thickness");
    const auto& parsePointChargeThicknessFunc =
        funcMap.at("point_charge_thickness");

    parseSmoothingRegionThicknessFunc(
        {"smoothing_region_thickness", "=", "1.25"},
        0
    );
    EXPECT_DOUBLE_EQ(
        settings::HybridSettings::getSmoothingRegionThickness(),
        1.25
    );

    _clearParser(parser);

    parsePointChargeThicknessFunc({"point_charge_thickness", "=", "4.75"}, 0);
    EXPECT_DOUBLE_EQ(settings::HybridSettings::getPointChargeThickness(), 4.75);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseSmoothingRegionThicknessFunc(
            {"smoothing_region_thickness", "=", "-0.1"},
            0
        ),
        exc::InputFileException,
        "Invalid value \"-0.1\" for key \"smoothing_region_thickness\" at line "
        "0 in input file: failed validation with message Value must be greater "
        "than or equal to 0"
    )

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parsePointChargeThicknessFunc(
            {"point_charge_thickness", "=", "-0.5"},
            0
        ),
        exc::InputFileException,
        "Invalid value \"-0.5\" for key \"point_charge_thickness\" at line 0 "
        "in input file: failed validation with message Value must be greater "
        "than or equal to 0"
    )
}

TEST_F(TestInputFileReader, parseSmoothingMethod)
{
    using enum SmoothingMethod;

    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("smoothing_method"));
    const auto& parseSmoothingMethodFunc = funcMap.at("smoothing_method");

    parseSmoothingMethodFunc({"smoothing_method", "=", "hotspot"}, 0);
    EXPECT_EQ(settings::HybridSettings::getSmoothingMethod(), HOTSPOT);

    _clearParser(parser);

    parseSmoothingMethodFunc({"smoothing_method", "=", "exact"}, 0);
    EXPECT_EQ(settings::HybridSettings::getSmoothingMethod(), EXACT);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseSmoothingMethodFunc({"smoothing_method", "=", "invalid"}, 0),
        exc::InputFileException,
        "Invalid value \"invalid\" for key \"smoothing_method\" at line 0 in "
        "input file. Allowed values: hotspot, exact"
    )
}

TEST_F(TestInputFileReader, parseQMForceDistribution)
{
    using enum QMForceDist;

    auto       parser  = input::HybridInputParser{};
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("qm_force_distribution"));
    const auto& parseQMForceDistributionFunc =
        funcMap.at("qm_force_distribution");

    parseQMForceDistributionFunc({"qm_force_distribution", "=", "none"}, 0);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), NONE);

    _clearParser(parser);

    parseQMForceDistributionFunc({"qm_force_distribution", "=", "equal"}, 0);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), EQUAL);

    _clearParser(parser);

    parseQMForceDistributionFunc({"qm_force_distribution", "=", "random"}, 0);
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), RANDOM);

    _clearParser(parser);

    parseQMForceDistributionFunc(
        {"qm_force_distribution", "=", "distance-weighted"},
        0
    );
    EXPECT_EQ(settings::HybridSettings::getQMForceDist(), DISTANCE_WEIGHTED);

    _clearParser(parser);

    ASSERT_THROW_MSG(
        parseQMForceDistributionFunc(
            {"qm_force_distribution", "=", "invalid"},
            0
        ),
        exc::InputFileException,
        "Invalid value \"invalid\" for key \"qm_force_distribution\" at line 0 "
        "in input file. Allowed values: none, equal, random, distance_weighted"
    )
}

TEST_F(TestInputFileReader, parseSelection)
{
    auto converter = input::Converter<input::SelectionTag>(
        "5, 3-4, 4, 1",   // gets ignored here
        "forced_inner_list"
    );

    EXPECT_EQ(
        converter.tryParse("5,3-4,4,1").value().indices,
        std::vector<int>({1, 3, 4, 5})
    );

    EXPECT_EQ(converter.tryParse("").value().indices, std::vector<int>({0}));
}

TEST_F(TestInputFileReader, parseSelectionNoPython)
{
    auto converter = input::Converter<input::SelectionTag>(
        "7-8, 10, 12",   // gets ignored here
        "inner_region_center"
    );

    EXPECT_EQ(
        _parseSelectionNoPython(converter, " 7 - 8 , 10, 12 ").value(),
        std::vector<int>({7, 8, 10, 12})
    );

    ASSERT_EQ(_parseSelectionNoPython(converter, ","), std::nullopt);
    ASSERT_EQ(
        converter.describeDomain({}),
        "An atom index in the selection is invalid. Must be a valid integer."
    );

    ASSERT_EQ(_parseSelectionNoPython(converter, "1-a"), std::nullopt);
    ASSERT_EQ(
        converter.describeDomain({}),
        "The end index of the selection is invalid. Must be a valid integer."
    );
}
