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

#include <memory>   // for allocator

#include "exceptions.hpp"         // for UserInputException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "qmSettings.hpp"         // for QMSettings, QMMethod
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

using namespace settings;
using namespace customException;

TEST(QMSettingsTest, SetQMMethodTest)
{
    QMSettings::setQMMethod("dftbplus");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::DFTBPLUS);

    QMSettings::setQMMethod("pyscf");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::PYSCF);

    QMSettings::setQMMethod("turbomole");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::TURBOMOLE);

    QMSettings::setQMMethod("mace");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::MACE);

    QMSettings::setQMMethod("ase_dftbplus");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::ASEDFTBPLUS);

    QMSettings::setQMMethod("ase_xtb");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::ASEXTB);

    QMSettings::setQMMethod("none");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::NONE);
}

TEST(QMSettingsTest, SetMaceModelSizeTest)
{
    QMSettings::setMaceModelSize("small");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::SMALL);
    
    QMSettings::setMaceModelSize("medium");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUM);

    QMSettings::setMaceModelSize("large");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::LARGE);

    QMSettings::setMaceModelSize("small-0b");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::SMALL0B);

    QMSettings::setMaceModelSize("medium-0b");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUM0B);

    QMSettings::setMaceModelSize("small-0b2");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::SMALL0B2);

    QMSettings::setMaceModelSize("medium-0b2");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUM0B2);

    QMSettings::setMaceModelSize("large-0b2");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::LARGE0B2);

    QMSettings::setMaceModelSize("medium-0b3");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUM0B3);

    QMSettings::setMaceModelSize("medium-mpa-0");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUMMPA0);

    QMSettings::setMaceModelSize("medium-omat-0");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUMOMAT0);

    ASSERT_THROW_MSG(
        QMSettings::setMaceModelSize("notAMaceModelSize"),
        UserInputException,
        "Mace model size notAMaceModelSize not recognized"
    );
}

TEST(QMSettingsTest, SetMaceModelTypeTest)
{
    QMSettings::setMaceModelType("mace_mp");
    EXPECT_EQ(QMSettings::getMaceModelType(), MaceModelType::MACE_MP);

    QMSettings::setMaceModelType("mace_off");
    EXPECT_EQ(QMSettings::getMaceModelType(), MaceModelType::MACE_OFF);

    QMSettings::setMaceModelType("mace_anicc");
    EXPECT_EQ(QMSettings::getMaceModelType(), MaceModelType::MACE_ANICC);

    ASSERT_THROW_MSG(
        QMSettings::setMaceModelType("notAMaceModelType"),
        UserInputException,
        "Mace notAMaceModelType model not recognized"
    )
}

TEST(QMSettingsTest, SetSlakosTypeTest)
{
    QMSettings::setSlakosType("3ob");
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::THREEOB);

    QMSettings::setSlakosType("matsci");
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::MATSCI);

    QMSettings::setSlakosType("custom");
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::CUSTOM);

    QMSettings::setSlakosType("none");
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::NONE);

    QMSettings::setSlakosType(SlakosType::THREEOB);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::THREEOB);

    QMSettings::setSlakosType(SlakosType::MATSCI);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::MATSCI);

    QMSettings::setSlakosType(SlakosType::CUSTOM);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::CUSTOM);

    QMSettings::setSlakosType(SlakosType::NONE);
    EXPECT_EQ(QMSettings::getSlakosType(), SlakosType::NONE);

    ASSERT_THROW_MSG(
        QMSettings::setSlakosType("notASlakosType"),
        UserInputException,
        "Slakos notASlakosType not recognized"
    );
}

TEST(QMSettingsTest, SetSlakosPathTest)
{
    QMSettings::setSlakosType("none");
    ASSERT_THROW_MSG(
        QMSettings::setSlakosPath("/path/to/slakos"),
        UserInputException,
        "Slakos path cannot be set without a slakos type"
    );

    QMSettings::setSlakosType("custom");
    QMSettings::setSlakosPath("/path/to/slakos");
    EXPECT_EQ(QMSettings::getSlakosPath(), "/path/to/slakos");

    QMSettings::setSlakosType("3ob");
    ASSERT_THROW_MSG(
        QMSettings::setSlakosPath("/path/to/slakos"),
        UserInputException,
        "Slakos path cannot be set for slakos type: 3ob"
    );

    QMSettings::setSlakosType("matsci");
    ASSERT_THROW_MSG(
        QMSettings::setSlakosPath("/path/to/slakos"),
        UserInputException,
        "Slakos path cannot be set for slakos type: matsci"
    );
}

TEST(QMSettingsTest, SetXtbMethodTest)
{
    QMSettings::setXtbMethod("GFN1-XtB");
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN1);

    QMSettings::setXtbMethod("gFn2_xTb");
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN2);

    QMSettings::setXtbMethod("IpeA1-xtB");
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::IPEA1);

    QMSettings::setXtbMethod(XtbMethod::GFN1);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN1);

    QMSettings::setXtbMethod(XtbMethod::GFN2);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::GFN2);

    QMSettings::setXtbMethod(XtbMethod::IPEA1);
    EXPECT_EQ(QMSettings::getXtbMethod(), XtbMethod::IPEA1);

    ASSERT_THROW_MSG(
        QMSettings::setXtbMethod("notAnXtbMethod"),
        UserInputException,
        "xTB method \"notAnXtbMethod\" not recognized"
    );
}

TEST(QMSettingsTest, ReturnQMMethodTest)
{
    EXPECT_EQ(string(QMMethod::DFTBPLUS), "DFTBPLUS");
    EXPECT_EQ(string(QMMethod::ASEDFTBPLUS), "ASEDFTBPLUS");
    EXPECT_EQ(string(QMMethod::ASEXTB), "ASEXTB");
    EXPECT_EQ(string(QMMethod::PYSCF), "PYSCF");
    EXPECT_EQ(string(QMMethod::TURBOMOLE), "TURBOMOLE");
    EXPECT_EQ(string(QMMethod::MACE), "MACE");
    EXPECT_EQ(string(QMMethod::NONE), "none");
}

TEST(QMSettingsTest, ReturnSlakosTypeTest)
{
    EXPECT_EQ(string(SlakosType::THREEOB), "3ob");
    EXPECT_EQ(string(SlakosType::MATSCI), "matsci");
    EXPECT_EQ(string(SlakosType::CUSTOM), "custom");
    EXPECT_EQ(string(SlakosType::NONE), "none");
}

TEST(QMSettingsTest, ReturnMaceModelTypeTest)
{
    EXPECT_EQ(string(MaceModelType::MACE_MP), "mace_mp");
    EXPECT_EQ(string(MaceModelType::MACE_OFF), "mace_off");
    EXPECT_EQ(string(MaceModelType::MACE_ANICC), "mace_anicc");
}

TEST(QMSettingsTest, ReturnMaceModelSizeTest)
{
    EXPECT_EQ(string(MaceModelSize::SMALL), "small");
    EXPECT_EQ(string(MaceModelSize::MEDIUM), "medium");
    EXPECT_EQ(string(MaceModelSize::LARGE), "large");
    EXPECT_EQ(string(MaceModelSize::SMALL0B), "small-0b");
    EXPECT_EQ(string(MaceModelSize::MEDIUM0B), "medium-0b");
    EXPECT_EQ(string(MaceModelSize::SMALL0B2), "small-0b2");
    EXPECT_EQ(string(MaceModelSize::MEDIUM0B2), "medium-0b2");
    EXPECT_EQ(string(MaceModelSize::LARGE0B2), "large-0b2");
    EXPECT_EQ(string(MaceModelSize::MEDIUM0B3), "medium-0b3");
    EXPECT_EQ(string(MaceModelSize::MEDIUMMPA0), "medium-mpa-0");
    EXPECT_EQ(string(MaceModelSize::MEDIUMOMAT0), "medium-omat-0");
}

TEST(QMSettingsTest, ReturnXtbMethodTest)
{
    EXPECT_EQ(string(XtbMethod::GFN1), "GFN1-xTB");
    EXPECT_EQ(string(XtbMethod::GFN2), "GFN2-xTB");
    EXPECT_EQ(string(XtbMethod::IPEA1), "IPEA1-xTB");
}