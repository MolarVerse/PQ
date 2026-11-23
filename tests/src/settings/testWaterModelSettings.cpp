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
#include <cmath>
#include <optional>
#include <string_view>
#include <utility>

#include "referenceSettings.hpp"
#include "units.hpp"
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

namespace
{

    /**
     * @brief H-O-H angle in degrees of a rigid water geometry
     */
    Angle hohAngle(const RigidWaterGeometry &geometry)
    {
        return 2.0 *
               ::asin(
                   0.5 * (geometry.dHH / geometry.dOH).numerical_value_in(one)
               ) *
               rad;
    }

    std::optional<RigidWaterGeometry> geometryOf(WaterIntraModel model)
    {
        WaterModelSettings settings;
        settings.setWaterIntraModelByUser(model);
        return settings.getRigidWaterGeometry();
    }
}   // namespace

/**
 * @brief without an intramolecular water model there is no rigid geometry
 */
TEST(TestWaterModelSettings, RigidGeometryIsEmptyIfNoModelIsSet)
{
    const WaterModelSettings settings;

    EXPECT_FALSE(settings.getRigidWaterGeometry().has_value());
}

/**
 * @brief every rigid model gives its published O-H distance and H-O-H angle
 */
TEST(TestWaterModelSettings, RigidGeometryMatchesTheModels)
{
    struct Case
    {
        WaterIntraModel model;
        Pos             dOH;
        Angle           angle;
    };

    constexpr Angle tetrahedral = 109.47122063449069 * deg;
    constexpr Angle tip3pAngle  = 104.52 * deg;

    const std::array cases = {
        Case{
            .model = WaterIntraModel::SPC,
            .dOH   = 1.0 * angstrom,
            .angle = tetrahedral
        },
        Case{
            .model = WaterIntraModel::SPC_E,
            .dOH   = 1.0 * angstrom,
            .angle = tetrahedral
        },
        Case{
            .model = WaterIntraModel::SPC_DC,
            .dOH   = 1.0 * angstrom,
            .angle = tetrahedral
        },
        Case{
            .model = WaterIntraModel::H2O_DC,
            .dOH   = 0.958 * angstrom,
            .angle = tetrahedral
        },
        Case{
            .model = WaterIntraModel::TIP3P,
            .dOH   = 0.9572 * angstrom,
            .angle = tip3pAngle
        },
        Case{
            .model = WaterIntraModel::OPC3,
            .dOH   = 0.97888 * angstrom,
            .angle = 109.47 * deg
        },
    };

    for (const auto &testCase : cases)
    {
        const auto geometry = geometryOf(testCase.model);

        ASSERT_TRUE(geometry.has_value())
            << WaterIntraModelMeta::toString(testCase.model);
        EXPECT_NEAR(
            geometry->dOH.numerical_value_in(angstrom),
            testCase.dOH.numerical_value_in(angstrom),
            1e-9
        ) << WaterIntraModelMeta::toString(testCase.model);
        EXPECT_NEAR(
            hohAngle(*geometry).numerical_value_in(deg),
            testCase.angle.numerical_value_in(deg),
            1e-3
        ) << WaterIntraModelMeta::toString(testCase.model);
    }
}

/**
 * @brief flexible models have no rigid geometry
 */
TEST(TestWaterModelSettings, FlexibleModelsHaveNoRigidGeometry)
{
    for (const auto model :
         {WaterIntraModel::SPC_FW,
          WaterIntraModel::QSPC_FW,
          WaterIntraModel::SPC_MTR,
          WaterIntraModel::TIP3P_MTR,
          WaterIntraModel::NONE})
    {
        EXPECT_FALSE(geometryOf(model).has_value())
            << WaterIntraModelMeta::toString(model);
    }
}

/**
 * @brief finalize() adds the reference file of the intramolecular model
 */
TEST(TestWaterModelSettings, FinalizeAddsReferenceOfIntraModel)
{
    const std::array cases = {
        std::pair{WaterIntraModel::SPC, ReferenceFiles::SPC_FILE},
        std::pair{WaterIntraModel::SPC_E, ReferenceFiles::SPC_E_FILE},
        std::pair{WaterIntraModel::SPC_FW, ReferenceFiles::SPC_FW_FILE},
        std::pair{WaterIntraModel::QSPC_FW, ReferenceFiles::QSPC_FW_FILE},
        std::pair{WaterIntraModel::SPC_DC, ReferenceFiles::SPC_DC_FILE},
        std::pair{WaterIntraModel::H2O_DC, ReferenceFiles::H2O_DC_FILE},
        std::pair{WaterIntraModel::TIP3P, ReferenceFiles::TIP3P_FILE},
        std::pair{WaterIntraModel::OPC3, ReferenceFiles::OPC3_FILE},
        std::pair{WaterIntraModel::SPC_MTR, ReferenceFiles::SPC_MTR_FILE},
        std::pair{WaterIntraModel::TIP3P_MTR, ReferenceFiles::TIP3P_MTR_FILE},
    };

    for (const auto &[model, file] : cases)
    {
        WaterModelSettings settings;
        settings.setWaterIntraModelByUser(model);
        ReferenceSettings references;
        settings.finalize(references);

        EXPECT_EQ(references.getReferenceFileNames().size(), 1)
            << WaterIntraModelMeta::toString(model);
        EXPECT_TRUE(references.getReferenceFileNames().contains(file))
            << WaterIntraModelMeta::toString(model);
    }
}

/**
 * @brief finalize() adds the reference file of the intermolecular model
 */
TEST(TestWaterModelSettings, FinalizeAddsReferenceOfInterModel)
{
    const std::array cases = {
        std::pair{WaterInterModel::SPC, ReferenceFiles::SPC_FILE},
        std::pair{WaterInterModel::SPC_E, ReferenceFiles::SPC_E_FILE},
        std::pair{WaterInterModel::SPC_FW, ReferenceFiles::SPC_FW_FILE},
        std::pair{WaterInterModel::QSPC_FW, ReferenceFiles::QSPC_FW_FILE},
        std::pair{WaterInterModel::SPC_DC, ReferenceFiles::SPC_DC_FILE},
        std::pair{WaterInterModel::H2O_DC, ReferenceFiles::H2O_DC_FILE},
        std::pair{WaterInterModel::TIP3P, ReferenceFiles::TIP3P_FILE},
        std::pair{WaterInterModel::OPC3, ReferenceFiles::OPC3_FILE},
        std::pair{WaterInterModel::SPC_MTR, ReferenceFiles::SPC_MTR_FILE},
        std::pair{WaterInterModel::TIP3P_MTR, ReferenceFiles::TIP3P_MTR_FILE},
    };

    for (const auto &[model, file] : cases)
    {
        WaterModelSettings settings;
        settings.setWaterInterModelByUser(model);
        ReferenceSettings references;
        settings.finalize(references);

        EXPECT_EQ(references.getReferenceFileNames().size(), 1)
            << WaterInterModelMeta::toString(model);
        EXPECT_TRUE(references.getReferenceFileNames().contains(file))
            << WaterInterModelMeta::toString(model);
    }
}

/**
 * @brief the same model for both parts adds its reference file only once, and
 * different models add both
 */
TEST(TestWaterModelSettings, FinalizeCombinesIntraAndInterReferences)
{
    WaterModelSettings same;
    same.setWaterIntraModelByUser(WaterIntraModel::TIP3P);
    same.setWaterInterModelByUser(WaterInterModel::TIP3P);
    ReferenceSettings sameReferences;
    same.finalize(sameReferences);
    EXPECT_EQ(sameReferences.getReferenceFileNames().size(), 1);

    WaterModelSettings different;
    different.setWaterIntraModelByUser(WaterIntraModel::SPC_FW);
    different.setWaterInterModelByUser(WaterInterModel::TIP3P);
    ReferenceSettings differentReferences;
    different.finalize(differentReferences);
    EXPECT_EQ(differentReferences.getReferenceFileNames().size(), 2);
}
