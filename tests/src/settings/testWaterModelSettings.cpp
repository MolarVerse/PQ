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
#include "throwWithMessage.hpp"
#include "waterModelSettings.hpp"

TEST(TestWaterModelSettings, FlagsAndEnumSettersRoundTrip)
{
    settings::WaterModelSettings::setIsWaterModelSet(true);
    settings::WaterModelSettings::setIsInterWaterModelSet(true);
    EXPECT_TRUE(settings::WaterModelSettings::isWaterModelSet());
    EXPECT_TRUE(settings::WaterModelSettings::isInterWaterModelSet());

    settings::WaterModelSettings::setWaterIntraModel(WaterIntraModel::SPC);
    settings::WaterModelSettings::setWaterInterModel(WaterInterModel::NONE);
    EXPECT_EQ(
        settings::WaterModelSettings::getWaterIntraModel(),
        WaterIntraModel::SPC
    );
    EXPECT_EQ(
        settings::WaterModelSettings::getWaterInterModel(),
        WaterInterModel::NONE
    );

    settings::WaterModelSettings::setIsWaterModelSet(false);
    settings::WaterModelSettings::setIsInterWaterModelSet(false);
    EXPECT_FALSE(settings::WaterModelSettings::isWaterModelSet());
    EXPECT_FALSE(settings::WaterModelSettings::isInterWaterModelSet());
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
            .display = "SPC_E"
        },
        ModelCase{
            .input   = "SPC_FW",
            .model   = WaterIntraModel::SPC_FW,
            .display = "SPC_FW"
        },
        ModelCase{
            .input   = "qspc-fw",
            .model   = WaterIntraModel::QSPC_FW,
            .display = "QSPC_FW"
        },
        ModelCase{
            .input   = "spc-dc",
            .model   = WaterIntraModel::SPC_DC,
            .display = "SPC_DC"
        },
        ModelCase{
            .input   = "h2o-dc",
            .model   = WaterIntraModel::H2O_DC,
            .display = "H2O_DC"
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
            .display = "SPC_MTR"
        },
        ModelCase{
            .input   = "tip3p-mtr",
            .model   = WaterIntraModel::TIP3P_MTR,
            .display = "TIP3P_MTR"
        },
    };

    for (const auto &testCase : cases)
    {
        settings::WaterModelSettings::setWaterIntraModel(testCase.input);
        EXPECT_EQ(
            settings::WaterModelSettings::getWaterIntraModel(),
            testCase.model
        );
        EXPECT_EQ(
            WaterIntraModelMeta::toString(testCase.model),
            testCase.display
        );
    }

    EXPECT_EQ(WaterIntraModelMeta::toString(WaterIntraModel::SPC), "SPC");
    EXPECT_EQ(WaterIntraModelMeta::toString(WaterIntraModel::NONE), "NONE");
    EXPECT_THROW_MSG(
        settings::WaterModelSettings::setWaterIntraModel("unknown"),
        exc::UserInputException,
        "Water intra model \"unknown\" not recognized"
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
            .display = "SPC_E"
        },
        ModelCase{
            .input   = "SPC_FW",
            .model   = WaterInterModel::SPC_FW,
            .display = "SPC_FW"
        },
        ModelCase{
            .input   = "qspc-fw",
            .model   = WaterInterModel::QSPC_FW,
            .display = "QSPC_FW"
        },
        ModelCase{
            .input   = "spc-dc",
            .model   = WaterInterModel::SPC_DC,
            .display = "SPC_DC"
        },
        ModelCase{
            .input   = "h2o-dc",
            .model   = WaterInterModel::H2O_DC,
            .display = "H2O_DC"
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
            .display = "SPC_MTR"
        },
        ModelCase{
            .input   = "tip3p-mtr",
            .model   = WaterInterModel::TIP3P_MTR,
            .display = "TIP3P_MTR"
        },
    };

    for (const auto &testCase : cases)
    {
        settings::WaterModelSettings::setWaterInterModel(testCase.input);
        EXPECT_EQ(
            settings::WaterModelSettings::getWaterInterModel(),
            testCase.model
        );
        EXPECT_EQ(
            WaterInterModelMeta::toString(testCase.model),
            testCase.display
        );
    }

    EXPECT_EQ(WaterInterModelMeta::toString(WaterInterModel::NONE), "NONE");
    EXPECT_THROW_MSG(
        settings::WaterModelSettings::setWaterInterModel("unknown"),
        exc::UserInputException,
        "Water inter model \"unknown\" not recognized"
    );
}
