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

using exc::UserInputException;
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
        ModelCase{
            .input   = "spc-e",
            .model   = WaterIntraModel::SPC_E,
            .display = "SPC/E"
        },
        ModelCase{
            .input   = "SPC_FW",
            .model   = WaterIntraModel::SPC_FW,
            .display = "SPC/Fw"
        },
        ModelCase{
            .input   = "qspc-fw",
            .model   = WaterIntraModel::QSPC_FW,
            .display = "qSPC/Fw"
        },
        ModelCase{
            .input   = "spc-dc",
            .model   = WaterIntraModel::SPC_DC,
            .display = "SPC/DC"
        },
        ModelCase{
            .input   = "h2o-dc",
            .model   = WaterIntraModel::H2O_DC,
            .display = "H2O-DC"
        },
        ModelCase{
            .input   = "tip3p",
            .model   = WaterIntraModel::TIP3P,
            .display = "TIP3P"
        },
        ModelCase{
            .input   = "opc3",
            .model   = WaterIntraModel::OPC3,
            .display = "OPC3"
        },
        ModelCase{
            .input   = "spc-mtr",
            .model   = WaterIntraModel::SPC_MTR,
            .display = "SPC-mTR"
        },
        ModelCase{
            .input   = "tip3p-mtr",
            .model   = WaterIntraModel::TIP3P_MTR,
            .display = "TIP3P-mTR"
        },
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
        ModelCase{
            .input   = "spc",
            .model   = WaterInterModel::SPC,
            .display = "SPC"
        },
        ModelCase{
            .input   = "spc-e",
            .model   = WaterInterModel::SPC_E,
            .display = "SPC/E"
        },
        ModelCase{
            .input   = "SPC_FW",
            .model   = WaterInterModel::SPC_FW,
            .display = "SPC/Fw"
        },
        ModelCase{
            .input   = "qspc-fw",
            .model   = WaterInterModel::QSPC_FW,
            .display = "qSPC/Fw"
        },
        ModelCase{
            .input   = "spc-dc",
            .model   = WaterInterModel::SPC_DC,
            .display = "SPC/DC"
        },
        ModelCase{
            .input   = "h2o-dc",
            .model   = WaterInterModel::H2O_DC,
            .display = "H2O-DC"
        },
        ModelCase{
            .input   = "tip3p",
            .model   = WaterInterModel::TIP3P,
            .display = "TIP3P"
        },
        ModelCase{
            .input   = "opc3",
            .model   = WaterInterModel::OPC3,
            .display = "OPC3"
        },
        ModelCase{
            .input   = "spc-mtr",
            .model   = WaterInterModel::SPC_MTR,
            .display = "SPC-mTR"
        },
        ModelCase{
            .input   = "tip3p-mtr",
            .model   = WaterInterModel::TIP3P_MTR,
            .display = "TIP3P-mTR"
        },
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
