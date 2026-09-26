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

#include <gtest/gtest.h>   // for Test, InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ

#include <cstdlib>
#include <filesystem>

#include "exceptions.hpp"   // for exc::UserInputException
                            // for Message, TestPartResult
#include "qmSettings.hpp"   // for settings::QMSettings, settings::QMMethod
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

TEST(QMSettingsTest, SetQMMethodTest)
{
    settings::QMSettings::setQMMethod("dftbplus");
    EXPECT_EQ(
        settings::QMSettings::getQMMethod(),
        settings::QMMethod::DFTBPLUS
    );

    settings::QMSettings::setQMMethod("pyscf");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::PYSCF);

    settings::QMSettings::setQMMethod("turbomole");
    EXPECT_EQ(
        settings::QMSettings::getQMMethod(),
        settings::QMMethod::TURBOMOLE
    );

    settings::QMSettings::setQMMethod("mace");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::MACE);

    settings::QMSettings::setQMMethod("ase_dftbplus");
    EXPECT_EQ(
        settings::QMSettings::getQMMethod(),
        settings::QMMethod::ASEDFTBPLUS
    );

    settings::QMSettings::setQMMethod("ase_xtb");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::ASEXTB);

    settings::QMSettings::setQMMethod("none");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::NONE);
}

TEST(QMSettingsTest, SetMaceModelTest)
{
    settings::QMSettings::setMaceModel("small");
    EXPECT_EQ(settings::QMSettings::getMaceModel(), settings::MaceModel::SMALL);

    settings::QMSettings::setMaceModel("medium");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUM
    );

    settings::QMSettings::setMaceModel("large");
    EXPECT_EQ(settings::QMSettings::getMaceModel(), settings::MaceModel::LARGE);

    settings::QMSettings::setMaceModel("small-0b");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::SMALL0B
    );

    settings::QMSettings::setMaceModel("medium-0b");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUM0B
    );

    settings::QMSettings::setMaceModel("small-0b2");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::SMALL0B2
    );

    settings::QMSettings::setMaceModel("medium-0b2");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUM0B2
    );

    settings::QMSettings::setMaceModel("large-0b2");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::LARGE0B2
    );

    settings::QMSettings::setMaceModel("medium-0b3");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUM0B3
    );

    settings::QMSettings::setMaceModel("medium-mpa-0");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUMMPA0
    );

    settings::QMSettings::setMaceModel("medium-omat-0");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::MEDIUMOMAT0
    );

    settings::QMSettings::setMaceModel("custom");
    EXPECT_EQ(
        settings::QMSettings::getMaceModel(),
        settings::MaceModel::CUSTOM
    );

    ASSERT_THROW_MSG(
        settings::QMSettings::setMaceModel("notAMaceModel"),
        exc::UserInputException,
        "Mace model size notAMaceModel not recognized"
    );
}

TEST(QMSettingsTest, SetMaceModeTest)
{
    using enum settings::MaceMode;

    settings::QMSettings::setMaceMode("accurate");
    EXPECT_EQ(settings::QMSettings::getMaceMode(), ACCURATE);

    settings::QMSettings::setMaceMode("fast");
    EXPECT_EQ(settings::QMSettings::getMaceMode(), FAST);

    settings::QMSettings::setMaceMode(ACCURATE);
    EXPECT_EQ(settings::QMSettings::getMaceMode(), ACCURATE);

    EXPECT_EQ(string(ACCURATE), "accurate");
    EXPECT_EQ(string(FAST), "fast");

    ASSERT_THROW_MSG(
        settings::QMSettings::setMaceMode("notAMode"),
        exc::UserInputException,
        "Unknown mace_mode \"notAMode\". Valid values are \"accurate\" (exact "
        "e3nn reference) or \"fast\" (cuequivariance-accelerated)."
    );
}

TEST(QMSettingsTest, SetMaceModelTypeTest)
{
    settings::QMSettings::setMaceModelType("mace_mp");
    EXPECT_EQ(
        settings::QMSettings::getMaceModelType(),
        settings::MaceModelType::MACE_MP
    );

    settings::QMSettings::setMaceModelType("mace_off");
    EXPECT_EQ(
        settings::QMSettings::getMaceModelType(),
        settings::MaceModelType::MACE_OFF
    );

    settings::QMSettings::setMaceModelType("mace_anicc");
    EXPECT_EQ(
        settings::QMSettings::getMaceModelType(),
        settings::MaceModelType::MACE_ANICC
    );

    ASSERT_THROW_MSG(
        settings::QMSettings::setMaceModelType("notAMaceModelType"),
        exc::UserInputException,
        "Mace notAMaceModelType model not recognized"
    )
}

TEST(QMSettingsTest, SetSlakosTypeTest)
{
#ifdef WITH_ASE
    settings::QMSettings::setSlakosType("3ob");
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::THREEOB
    );

    settings::QMSettings::setSlakosType("matsci");
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::MATSCI
    );

    settings::QMSettings::setSlakosType(settings::SlakosType::THREEOB);
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::THREEOB
    );

    settings::QMSettings::setSlakosType(settings::SlakosType::MATSCI);
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::MATSCI
    );
#endif

    settings::QMSettings::setSlakosType("custom");
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::CUSTOM
    );

    settings::QMSettings::setSlakosType("none");
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::NONE
    );

    settings::QMSettings::setSlakosType(settings::SlakosType::CUSTOM);
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::CUSTOM
    );

    settings::QMSettings::setSlakosType(settings::SlakosType::NONE);
    EXPECT_EQ(
        settings::QMSettings::getSlakosType(),
        settings::SlakosType::NONE
    );

    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosType("notASlakosType"),
        exc::UserInputException,
        "Slakos notASlakosType not recognized"
    );
}

#ifdef WITH_ASE
TEST(QMSettingsTest, ResolvesBundledSlakos)
{
    const auto *expectedRoot = std::getenv("PQ_TEST_EXPECTED_SLAKOS_ROOT");

    for (const auto *slakos : {"3ob", "matsci"})
    {
        settings::QMSettings::setSlakosType(slakos);
        const auto path = std::filesystem::weakly_canonical(
            settings::QMSettings::getSlakosPath()
        );

        EXPECT_TRUE(std::filesystem::is_directory(path));
        if (expectedRoot != nullptr)
        {
            EXPECT_EQ(
                path,
                std::filesystem::weakly_canonical(
                    std::filesystem::path(expectedRoot) / slakos / "skfiles"
                )
            );
        }
    }
}
#endif

#ifndef WITH_ASE
TEST(QMSettingsTest, SetBuiltInSlakosTypeRequiresAse)
{
    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosType("3ob"),
        InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosType("matsci"),
        InputFileException,
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );

    settings::QMSettings::setSlakosType("none");
}
#endif

TEST(QMSettingsTest, SetSlakosPathTest)
{
    settings::QMSettings::setSlakosType("none");
    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosPath("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set without a slakos type"
    );

    settings::QMSettings::setSlakosType("custom");
    settings::QMSettings::setSlakosPath("/path/to/slakos");
    EXPECT_EQ(settings::QMSettings::getSlakosPath(), "/path/to/slakos");

#ifdef WITH_ASE
    settings::QMSettings::setSlakosType("3ob");
    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosPath("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set for slakos type: 3ob"
    );

    settings::QMSettings::setSlakosType("matsci");
    ASSERT_THROW_MSG(
        settings::QMSettings::setSlakosPath("/path/to/slakos"),
        exc::UserInputException,
        "Slakos path cannot be set for slakos type: matsci"
    );
#endif
}

TEST(QMSettingsTest, SetXtbMethodTest)
{
    settings::QMSettings::setXtbMethod("GFN1-XtB");
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::GFN1);

    settings::QMSettings::setXtbMethod("gFn2_xTb");
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::GFN2);

    settings::QMSettings::setXtbMethod("IpeA1-xtB");
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::IPEA1);

    settings::QMSettings::setXtbMethod(settings::XtbMethod::GFN1);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::GFN1);

    settings::QMSettings::setXtbMethod(settings::XtbMethod::GFN2);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::GFN2);

    settings::QMSettings::setXtbMethod(settings::XtbMethod::IPEA1);
    EXPECT_EQ(settings::QMSettings::getXtbMethod(), settings::XtbMethod::IPEA1);

    ASSERT_THROW_MSG(
        settings::QMSettings::setXtbMethod("notAnXtbMethod"),
        exc::UserInputException,
        "xTB method \"notAnXtbMethod\" not recognized"
    );
}

TEST(QMSettingsTest, ReturnQMMethodTest)
{
    EXPECT_EQ(string(settings::QMMethod::DFTBPLUS), "DFTBPLUS");
    EXPECT_EQ(string(settings::QMMethod::ASEDFTBPLUS), "ASEDFTBPLUS");
    EXPECT_EQ(string(settings::QMMethod::ASEXTB), "ASEXTB");
    EXPECT_EQ(string(settings::QMMethod::PYSCF), "PYSCF");
    EXPECT_EQ(string(settings::QMMethod::TURBOMOLE), "TURBOMOLE");
    EXPECT_EQ(string(settings::QMMethod::MACE), "MACE");
    EXPECT_EQ(string(settings::QMMethod::FENNOL), "FeNNol");
    EXPECT_EQ(string(settings::QMMethod::NONE), "none");
}

TEST(QMSettingsTest, ReturnSlakosTypeTest)
{
    EXPECT_EQ(string(settings::SlakosType::THREEOB), "3ob");
    EXPECT_EQ(string(settings::SlakosType::MATSCI), "matsci");
    EXPECT_EQ(string(settings::SlakosType::CUSTOM), "custom");
    EXPECT_EQ(string(settings::SlakosType::NONE), "none");
}

TEST(QMSettingsTest, ReturnMaceModelTypeTest)
{
    EXPECT_EQ(string(settings::MaceModelType::MACE_MP), "mace_mp");
    EXPECT_EQ(string(settings::MaceModelType::MACE_OFF), "mace_off");
    EXPECT_EQ(string(settings::MaceModelType::MACE_ANICC), "mace_anicc");
}

TEST(QMSettingsTest, ReturnMaceModelTest)
{
    EXPECT_EQ(string(settings::MaceModel::SMALL), "small");
    EXPECT_EQ(string(settings::MaceModel::MEDIUM), "medium");
    EXPECT_EQ(string(settings::MaceModel::LARGE), "large");
    EXPECT_EQ(string(settings::MaceModel::SMALL0B), "small-0b");
    EXPECT_EQ(string(settings::MaceModel::MEDIUM0B), "medium-0b");
    EXPECT_EQ(string(settings::MaceModel::SMALL0B2), "small-0b2");
    EXPECT_EQ(string(settings::MaceModel::MEDIUM0B2), "medium-0b2");
    EXPECT_EQ(string(settings::MaceModel::LARGE0B2), "large-0b2");
    EXPECT_EQ(string(settings::MaceModel::MEDIUM0B3), "medium-0b3");
    EXPECT_EQ(string(settings::MaceModel::MEDIUMMPA0), "medium-mpa-0");
    EXPECT_EQ(string(settings::MaceModel::MEDIUMOMAT0), "medium-omat-0");
    EXPECT_EQ(string(settings::MaceModel::CUSTOM), "custom");
}

TEST(QMSettingsTest, ReturnMaceModeTest)
{
    EXPECT_EQ(string(settings::MaceMode::ACCURATE), "accurate");
    EXPECT_EQ(string(settings::MaceMode::FAST), "fast");
}

TEST(QMSettingsTest, ReturnXtbMethodTest)
{
    EXPECT_EQ(string(settings::XtbMethod::GFN1), "GFN1-xTB");
    EXPECT_EQ(string(settings::XtbMethod::GFN2), "GFN2-xTB");
    EXPECT_EQ(string(settings::XtbMethod::IPEA1), "IPEA1-xTB");
}

TEST(QMSettingsTest, SetFennolModelPath)
{
    settings::QMSettings::setFennolModelPath("/paTh/to/fennol_model.fnx");
    EXPECT_EQ(
        settings::QMSettings::getFennolModelPath(),
        "/paTh/to/fennol_model.fnx"
    );
}

TEST(QMSettingsTest, SetGPUPreprocessing)
{
    settings::QMSettings::setUseGPUPreprocessing(false);
    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), false);
    settings::QMSettings::setUseGPUPreprocessing(true);
    EXPECT_EQ(settings::QMSettings::useGPUPreprocessing(), true);
}
