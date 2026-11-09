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

#include <cstdlib>
#include <filesystem>

#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "qmSettings.hpp"
#include "throwWithMessage.hpp"

TEST(QMSettingsTest, SetMaceModelTypeTest)
{
    MaceSettings settings;
    settings.setMaceModelTypeByUser(MaceModelType::MACE_MP);
    EXPECT_EQ(settings.getMaceModelType(), MaceModelType::MACE_MP);

    settings.setMaceModelTypeByUser(MaceModelType::MACE_OFF);
    EXPECT_EQ(settings.getMaceModelType(), MaceModelType::MACE_OFF);

    settings.setMaceModelTypeByUser(MaceModelType::MACE_ANICC);
    EXPECT_EQ(settings.getMaceModelType(), MaceModelType::MACE_ANICC);
}

TEST(QMSettingsTest, SetSlakosTypeTest)
{
    DftbSettings dftbSettings;
#ifdef WITH_ASE
    dftbSettings.setSlakosTypeByUser(SlakosType::NONE);
    EXPECT_EQ(dftbSettings.getSlakosType(), SlakosType::NONE);

    dftbSettings.setSlakosTypeByUser(SlakosType::MATSCI);
    EXPECT_EQ(dftbSettings.getSlakosType(), SlakosType::MATSCI);

    dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB);
    EXPECT_EQ(dftbSettings.getSlakosType(), SlakosType::THREEOB);

#endif

    dftbSettings.setSlakosTypeByUser(SlakosType::CUSTOM);
    EXPECT_EQ(dftbSettings.getSlakosType(), SlakosType::CUSTOM);

    dftbSettings.setSlakosTypeByUser(SlakosType::NONE);
    EXPECT_EQ(dftbSettings.getSlakosType(), SlakosType::NONE);
}

#ifdef WITH_ASE
TEST(QMSettingsTest, ResolvesBundledSlakos)
{
    const auto *expectedRoot = std::getenv("PQ_TEST_EXPECTED_SLAKOS_ROOT");

    DftbSettings dftbSettings;

    for (const auto slakos : {SlakosType::THREEOB, SlakosType::MATSCI})
    {
        dftbSettings.setSlakosTypeByUser(slakos);
        const auto path =
            std::filesystem::weakly_canonical(dftbSettings.getSlakosPath());

        EXPECT_TRUE(std::filesystem::is_directory(path));
        if (expectedRoot != nullptr)
        {
            EXPECT_EQ(
                path,
                std::filesystem::weakly_canonical(
                    std::filesystem::path(expectedRoot) /
                    SlakosTypeMeta::toString(slakos) / "skfiles"
                )
            );
        }
    }
}
#endif

#ifndef WITH_ASE
TEST(QMSettingsTest, SetBuiltInSlakosTypeRequiresAse)
{
    DftbSettings dftbSettings;
    ASSERT_THROW_MSG(
        dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB),
        exc::InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    ASSERT_THROW_MSG(
        dftbSettings.setSlakosTypeByUser(SlakosType::MATSCI),
        exc::InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    dftbSettings.setSlakosTypeByUser(SlakosType::NONE);
}
#endif

TEST(QMSettingsTest, SetSlakosPathTest)
{
    DftbSettings dftbSettings;
    dftbSettings.setSlakosTypeByUser(SlakosType::NONE);
    ASSERT_THROW_MSG(
        dftbSettings.setSlakosPathByUser("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set without a slakos type"
    );

    dftbSettings.setSlakosTypeByUser(SlakosType::CUSTOM);
    dftbSettings.setSlakosPathByUser("/path/to/slakos");
    EXPECT_EQ(dftbSettings.getSlakosPath(), "/path/to/slakos");

#ifdef WITH_ASE
    dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB);
    ASSERT_THROW_MSG(
        dftbSettings.setSlakosPathByUser("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set for slakos type: 3ob"
    );

    dftbSettings.setSlakosTypeByUser(SlakosType::MATSCI);
    ASSERT_THROW_MSG(
        dftbSettings.setSlakosPathByUser("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set for slakos type: matsci"
    );
#endif
}

TEST(QMSettingsTest, SetXtbMethodTest)
{
    XtbSettings xtbSettings;
    xtbSettings.setXtbMethodByUser(XtbMethod::GFN1);
    EXPECT_EQ(xtbSettings.getXtbMethod(), XtbMethod::GFN1);

    xtbSettings.setXtbMethodByUser(XtbMethod::GFN2);
    EXPECT_EQ(xtbSettings.getXtbMethod(), XtbMethod::GFN2);

    xtbSettings.setXtbMethodByUser(XtbMethod::IPEA1);
    EXPECT_EQ(xtbSettings.getXtbMethod(), XtbMethod::IPEA1);
}

TEST(QMSettingsTest, ReturnQMMethodTest)
{
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::DFTBPLUS), "DFTBPLUS");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::ASE_DFTBPLUS), "ASE_DFTBPLUS");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::ASE_XTB), "ASE_XTB");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::PYSCF), "PYSCF");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::TURBOMOLE), "TURBOMOLE");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::ORCA), "ORCA");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::MACE), "MACE");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::FENNOL), "FENNOL");
    EXPECT_EQ(QMMethodMeta::toString(QMMethod::NONE), "NONE");
}

TEST(QMSettingsTest, ReturnSlakosTypeTest)
{
    EXPECT_EQ(SlakosTypeMeta::toString(SlakosType::THREEOB), "3ob");
    EXPECT_EQ(SlakosTypeMeta::toString(SlakosType::MATSCI), "matsci");
    EXPECT_EQ(SlakosTypeMeta::toString(SlakosType::CUSTOM), "custom");
    EXPECT_EQ(SlakosTypeMeta::toString(SlakosType::NONE), "none");
}

TEST(QMSettingsTest, ReturnMaceModelTypeTest)
{
    EXPECT_EQ(MaceModelTypeMeta::toString(MaceModelType::MACE_MP), "MACE_MP");
    EXPECT_EQ(MaceModelTypeMeta::toString(MaceModelType::MACE_OFF), "MACE_OFF");
    EXPECT_EQ(
        MaceModelTypeMeta::toString(MaceModelType::MACE_ANICC),
        "MACE_ANICC"
    );
}

TEST(QMSettingsTest, ReturnMaceModelTest)
{
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::SMALL), "SMALL");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::MEDIUM), "MEDIUM");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::LARGE), "LARGE");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::SMALL_0B), "SMALL_0B");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::MEDIUM_0B), "MEDIUM_0B");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::SMALL_0B2), "SMALL_0B2");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::MEDIUM_0B2), "MEDIUM_0B2");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::LARGE_0B2), "LARGE_0B2");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::MEDIUM_0B3), "MEDIUM_0B3");
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::MEDIUM_MPA_0), "MEDIUM_MPA_0");
    EXPECT_EQ(
        MaceModelMeta::toString(MaceModel::MEDIUM_OMAT_0),
        "MEDIUM_OMAT_0"
    );
    EXPECT_EQ(MaceModelMeta::toString(MaceModel::CUSTOM), "CUSTOM");
}

TEST(QMSettingsTest, ReturnMaceModeTest)
{
    EXPECT_EQ(MaceModeMeta::toString(MaceMode::ACCURATE), "ACCURATE");
    EXPECT_EQ(MaceModeMeta::toString(MaceMode::FAST), "FAST");
}

TEST(QMSettingsTest, ReturnXtbMethodTest)
{
    EXPECT_EQ(XtbMethodMeta::toString(XtbMethod::GFN1), "GFN1-xTB");
    EXPECT_EQ(XtbMethodMeta::toString(XtbMethod::GFN2), "GFN2-xTB");
    EXPECT_EQ(XtbMethodMeta::toString(XtbMethod::IPEA1), "IPEA1-xTB");
}

TEST(QMSettingsTest, SetFennolModelPath)
{
    FennolSettings fennolSettings;
    fennolSettings.setFennolModelPathByUser("/paTh/to/fennol_model.fnx");
    EXPECT_EQ(fennolSettings.getFennolModelPath(), "/paTh/to/fennol_model.fnx");
}

TEST(QMSettingsTest, SetGPUPreprocessing)
{
    FennolSettings fennolSettings;
    fennolSettings.setUseGPUPreprocessingByUser(false);
    EXPECT_EQ(fennolSettings.useGPUPreprocessing(), false);
    fennolSettings.setUseGPUPreprocessingByUser(true);
    EXPECT_EQ(fennolSettings.useGPUPreprocessing(), true);
}

TEST(QMSettingsTest, MaceNamesForMaceCalculatorsTest)
{
    MaceSettings settings;
    settings.setMaceModelTypeByUser(MaceModelType::MACE_OFF);
    EXPECT_EQ(settings.getMaceModelTypeName(), "mace_off");

    settings.setMaceModelTypeByUser(MaceModelType::MACE_MP);
    EXPECT_EQ(settings.getMaceModelTypeName(), "mace_mp");

    settings.setMaceModelByUser(MaceModel::SMALL);
    EXPECT_EQ(settings.getMaceModelName(), "small");

    settings.setMaceModelByUser(MaceModel::MEDIUM_0B3);
    EXPECT_EQ(settings.getMaceModelName(), "medium-0b3");

    settings.setMaceModelByUser(MaceModel::MEDIUM_OMAT_0);
    EXPECT_EQ(settings.getMaceModelName(), "medium-omat-0");
}
