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

    QMSettings::setQMMethod("none");
    EXPECT_EQ(QMSettings::getQMMethod(), QMMethod::NONE);
}

TEST(QMSettingsTest, SetMaceModelSizeTest)
{
    QMSettings::setMaceModelSize("large");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::LARGE);

    QMSettings::setMaceModelSize("medium");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::MEDIUM);

    QMSettings::setMaceModelSize("small");
    EXPECT_EQ(QMSettings::getMaceModelSize(), MaceModelSize::SMALL);

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

    ASSERT_THROW_MSG(
        QMSettings::setSlakosType("notASlakosType"),
        UserInputException,
        "Slakos notASlakosType not recognized"
    );
}

TEST(QMSettingsTest, ReturnQMMethodTest)
{
    EXPECT_EQ(string(QMMethod::DFTBPLUS), "DFTBPLUS");
    EXPECT_EQ(string(QMMethod::ASEDFTBPLUS), "ASEDFTBPLUS");
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