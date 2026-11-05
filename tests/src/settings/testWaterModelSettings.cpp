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

#include "referenceSettings.hpp"
#include "waterModelSettings.hpp"

TEST(TestWaterModelSettings, FlagsAndEnumSettersRoundTrip)
{
    WaterModelSettings settings;
    EXPECT_FALSE(settings.isWaterModelSetAndNotNone());
    EXPECT_FALSE(settings.isInterWaterModelSetAndNotNone());

    settings.setWaterIntraModelByUser(WaterIntraModel::SPC);
    settings.setWaterInterModelByUser(WaterInterModel::NONE);
    EXPECT_EQ(settings.getWaterIntraModel(), WaterIntraModel::SPC);
    EXPECT_EQ(settings.getWaterInterModel(), WaterInterModel::NONE);

    EXPECT_TRUE(settings.isWaterModelSetAndNotNone());
    EXPECT_FALSE(settings.isInterWaterModelSetAndNotNone());
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
        WaterModelSettings settings;
        settings.setWaterIntraModelByUser(testCase.model);
        EXPECT_EQ(settings.getWaterIntraModel(), testCase.model);
        EXPECT_EQ(
            WaterIntraModelMeta::toString(testCase.model),
            testCase.display
        );
    }

    EXPECT_EQ(WaterIntraModelMeta::toString(WaterIntraModel::SPC), "SPC");
    EXPECT_EQ(WaterIntraModelMeta::toString(WaterIntraModel::NONE), "NONE");
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
        WaterModelSettings settings;
        settings.setWaterInterModelByUser(testCase.model);
        EXPECT_EQ(settings.getWaterInterModel(), testCase.model);
        EXPECT_EQ(
            WaterInterModelMeta::toString(testCase.model),
            testCase.display
        );
    }

    EXPECT_EQ(WaterInterModelMeta::toString(WaterInterModel::NONE), "NONE");
}

/**
 * @brief finalize() must add a reference file for the intramolecular water
 * model even when no intermolecular model has been set - regression test
 * for a bug where the intra-model switch was gated on
 * isInterWaterModelSet() instead of isWaterModelSet(), silently dropping
 * the intra-model reference whenever only the intra model was configured.
 */
TEST(TestWaterModelSettings, FinalizeAddsIntraModelReferenceWithoutInterModel)
{
    WaterModelSettings settings;
    settings.setWaterIntraModelByUser(WaterIntraModel::SPC_FW);
    ASSERT_TRUE(settings.isWaterModelSetAndNotNone());
    ASSERT_FALSE(settings.isInterWaterModelSetAndNotNone());

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    EXPECT_TRUE(referenceSettings.getReferenceFileNames().contains(
        ReferenceFiles::SPC_FW_FILE
    ));
}

/**
 * @brief finalize() must add a reference file for the intermolecular water
 * model even when no intramolecular model has been set.
 */
TEST(TestWaterModelSettings, FinalizeAddsInterModelReferenceWithoutIntraModel)
{
    WaterModelSettings settings;
    settings.setWaterInterModelByUser(WaterInterModel::TIP3P);
    ASSERT_TRUE(settings.isInterWaterModelSetAndNotNone());

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    EXPECT_TRUE(referenceSettings.getReferenceFileNames().contains(
        ReferenceFiles::TIP3P_FILE
    ));
}

/**
 * @brief finalize() must add both references when both an intra- and an
 * inter-molecular model are set, even when they differ.
 */
TEST(TestWaterModelSettings, FinalizeAddsBothReferencesWhenBothModelsSet)
{
    WaterModelSettings settings;
    settings.setWaterIntraModelByUser(WaterIntraModel::SPC_MTR);
    settings.setWaterInterModelByUser(WaterInterModel::TIP3P_MTR);

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    const auto &referenceFiles = referenceSettings.getReferenceFileNames();
    EXPECT_TRUE(referenceFiles.contains(ReferenceFiles::SPC_MTR_FILE));
    EXPECT_TRUE(referenceFiles.contains(ReferenceFiles::TIP3P_MTR_FILE));
}

/**
 * @brief finalize() must not add any reference file when neither water
 * model has been set.
 */
TEST(TestWaterModelSettings, FinalizeAddsNoReferenceWhenNeitherModelSet)
{
    WaterModelSettings settings;

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    EXPECT_TRUE(referenceSettings.getReferenceFileNames().empty());
}
