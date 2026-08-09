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

#include <array>
#include <string_view>

#include "exceptions.hpp"
#include "waterModelSettings.hpp"

using customException::UserInputException;
using settings::WaterInterModel;
using settings::WaterIntraModel;
using settings::WaterModelSettings;

TEST(TestWaterModelSettings, FlagsAndEnumSettersRoundTrip)
{
    WaterModelSettings::setIsWaterModelSet(true);
    WaterModelSettings::setIsInterWaterModelSet(true);
    EXPECT_TRUE(WaterModelSettings::isWaterModelSet());
    EXPECT_TRUE(WaterModelSettings::isInterWaterModelSet());

    WaterModelSettings::setWaterIntraModel(WaterIntraModel::SPC);
    WaterModelSettings::setWaterInterModel(WaterInterModel::NONE);
    EXPECT_EQ(WaterModelSettings::getWaterIntraModel(), WaterIntraModel::SPC);
    EXPECT_EQ(WaterModelSettings::getWaterInterModel(), WaterInterModel::NONE);

    WaterModelSettings::setIsWaterModelSet(false);
    WaterModelSettings::setIsInterWaterModelSet(false);
    EXPECT_FALSE(WaterModelSettings::isWaterModelSet());
    EXPECT_FALSE(WaterModelSettings::isInterWaterModelSet());
}

TEST(TestWaterModelSettings, IntraModelNamesRoundTrip)
{
    struct ModelCase
    {
        std::string_view input;
        WaterIntraModel  model;
        std::string_view display;
    };

    constexpr std::array cases{
        ModelCase{"spc-e", WaterIntraModel::SPC_E, "SPC/E"},
        ModelCase{"SPC_FW", WaterIntraModel::SPC_FW, "SPC/Fw"},
        ModelCase{"qspc-fw", WaterIntraModel::QSPC_FW, "qSPC/Fw"},
        ModelCase{"spc-dc", WaterIntraModel::SPC_DC, "SPC/DC"},
        ModelCase{"h2o-dc", WaterIntraModel::H2O_DC, "H2O-DC"},
        ModelCase{"tip3p", WaterIntraModel::TIP3P, "TIP3P"},
        ModelCase{"opc3", WaterIntraModel::OPC3, "OPC3"},
        ModelCase{"spc-mtr", WaterIntraModel::SPC_MTR, "SPC-mTR"},
        ModelCase{"tip3p-mtr", WaterIntraModel::TIP3P_MTR, "TIP3P-mTR"},
    };

    for (const auto &testCase : cases)
    {
        WaterModelSettings::setWaterIntraModel(testCase.input);
        EXPECT_EQ(WaterModelSettings::getWaterIntraModel(), testCase.model);
        EXPECT_EQ(settings::string(testCase.model), testCase.display);
    }

    EXPECT_EQ(settings::string(WaterIntraModel::SPC), "SPC");
    EXPECT_EQ(settings::string(WaterIntraModel::NONE), "none");
    EXPECT_THROW(
        WaterModelSettings::setWaterIntraModel("unknown"),
        UserInputException
    );
}

TEST(TestWaterModelSettings, InterModelNamesRoundTrip)
{
    struct ModelCase
    {
        std::string_view input;
        WaterInterModel  model;
        std::string_view display;
    };

    constexpr std::array cases{
        ModelCase{"spc", WaterInterModel::SPC, "SPC"},
        ModelCase{"spc-e", WaterInterModel::SPC_E, "SPC/E"},
        ModelCase{"SPC_FW", WaterInterModel::SPC_FW, "SPC/Fw"},
        ModelCase{"qspc-fw", WaterInterModel::QSPC_FW, "qSPC/Fw"},
        ModelCase{"spc-dc", WaterInterModel::SPC_DC, "SPC/DC"},
        ModelCase{"h2o-dc", WaterInterModel::H2O_DC, "H2O-DC"},
        ModelCase{"tip3p", WaterInterModel::TIP3P, "TIP3P"},
        ModelCase{"opc3", WaterInterModel::OPC3, "OPC3"},
        ModelCase{"spc-mtr", WaterInterModel::SPC_MTR, "SPC-mTR"},
        ModelCase{"tip3p-mtr", WaterInterModel::TIP3P_MTR, "TIP3P-mTR"},
    };

    for (const auto &testCase : cases)
    {
        WaterModelSettings::setWaterInterModel(testCase.input);
        EXPECT_EQ(WaterModelSettings::getWaterInterModel(), testCase.model);
        EXPECT_EQ(settings::string(testCase.model), testCase.display);
    }

    EXPECT_EQ(settings::string(WaterInterModel::NONE), "none");
    EXPECT_THROW(
        WaterModelSettings::setWaterInterModel("unknown"),
        UserInputException
    );
}
