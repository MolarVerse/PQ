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

// #include <memory>   // for allocator

#include "settings.hpp"   // for Settings
// #include "exceptions.hpp"         // for UserInputException
// #include "gtest/gtest.h"          // for Message, TestPartResult
// #include "qmSettings.hpp"         // for QMSettings, QMMethod
// #include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

using enum settings::JobType;
using enum settings::FPType;

using namespace settings;
// using namespace customException;

TEST(TestSettings, string_JobtypeTest)
{
    EXPECT_EQ(string(MM_MD), "MM_MD");
    EXPECT_EQ(string(QM_MD), "QM_MD");
    EXPECT_EQ(string(QMMM_MD), "QMMM_MD");
    EXPECT_EQ(string(RING_POLYMER_QM_MD), "RING_POLYMER_QM_MD");
    EXPECT_EQ(string(MM_OPT), "MM_OPT");
    EXPECT_EQ(string(NONE), "NONE");
}

TEST(TestSettings, setJobtypeTest)
{
    Settings::setJobtype("MmMD");
    EXPECT_EQ(Settings::getJobtype(), MM_MD);

    Settings::setJobtype("qMMd");
    EXPECT_EQ(Settings::getJobtype(), QM_MD);

    Settings::setJobtype("RinG-POLymer_QMMd");
    EXPECT_EQ(Settings::getJobtype(), RING_POLYMER_QM_MD);

    Settings::setJobtype("qMmmMd");
    EXPECT_EQ(Settings::getJobtype(), QMMM_MD);

    Settings::setJobtype("MMoPT");
    EXPECT_EQ(Settings::getJobtype(), MM_OPT);

    Settings::setJobtype("not-a-jobtype");
    EXPECT_EQ(Settings::getJobtype(), NONE);

    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::getJobtype(), MM_MD);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::getJobtype(), QM_MD);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::getJobtype(), RING_POLYMER_QM_MD);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::getJobtype(), QMMM_MD);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::getJobtype(), MM_OPT);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::getJobtype(), NONE);
}

TEST(TestSettings, setFloatingPointTypeTest)
{
    Settings::setFloatingPointType("FlOAt");
    EXPECT_EQ(Settings::getFloatingPointType(), FLOAT);

    Settings::setFloatingPointType("DOUble");
    EXPECT_EQ(Settings::getFloatingPointType(), DOUBLE);

    Settings::setFloatingPointType("not-a-floating-point-type");
    EXPECT_EQ(Settings::getFloatingPointType(), DOUBLE);

    Settings::setFloatingPointType(FLOAT);
    EXPECT_EQ(Settings::getFloatingPointType(), FLOAT);

    Settings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(Settings::getFloatingPointType(), DOUBLE);

    Settings::setFloatingPointType(FLOAT);
    EXPECT_EQ(Settings::getFloatingPointPybindString(), "float32");

    Settings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(Settings::getFloatingPointPybindString(), "float64");
}

TEST(TestSettings, setRandomSeedTest)
{
    Settings::setRandomSeed(73);
    EXPECT_EQ(Settings::getRandomSeed(), 73);
}

TEST(TestSettings, setIsRandomSeedTest)
{
    Settings::setIsRandomSeedSet(true);
    EXPECT_EQ(Settings::isRandomSeedSet(), true);

    Settings::setIsRandomSeedSet(false);
    EXPECT_EQ(Settings::isRandomSeedSet(), false);
}

TEST(TestSettings, setIsRingPolymerMDActivatedTest)
{
    Settings::setIsRingPolymerMDActivated(true);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), true);

    Settings::setIsRingPolymerMDActivated(false);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);
}

TEST(TestSettings, setDimensionalityTest)
{
    Settings::setDimensionality(3);
    EXPECT_EQ(Settings::getDimensionality(), 3);
}

TEST(TestSettings, isQMOnlyJobtypeTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), true);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), true);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isQMOnlyJobtype(), false);
}

TEST(TestSettings, isMMOnlyJobtypeTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), true);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), false);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isMMOnlyJobtype(), false);
}

TEST(TestSettings, isHybridJobtypeTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isHybridJobtype(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isHybridJobtype(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isHybridJobtype(), false);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isHybridJobtype(), true);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isHybridJobtype(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isHybridJobtype(), false);
}

TEST(TestSettings, isMDJobtypeTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isMDJobType(), true);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isMDJobType(), true);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isMDJobType(), true);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isMDJobType(), true);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isMDJobType(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isMDJobType(), false);
}

TEST(TestSettings, isOptJobtypeTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isOptJobType(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isOptJobType(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isOptJobType(), false);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isOptJobType(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isOptJobType(), true);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isOptJobType(), false);
}

TEST(TestSettings, isMMActivatedTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isMMActivated(), true);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isMMActivated(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isMMActivated(), false);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isMMActivated(), true);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isMMActivated(), true);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isMMActivated(), false);
}

TEST(TestSettings, isQMActivatedTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isQMActivated(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isQMActivated(), true);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isQMActivated(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isQMActivated(), false);
}

TEST(TestSettings, isQMOnlyActivatedTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isQMOnlyActivated(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isQMOnlyActivated(), true);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isQMOnlyActivated(), true);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isQMOnlyActivated(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isQMOnlyActivated(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isQMOnlyActivated(), false);
}

TEST(TestSettings, isMMOnlyActivatedTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isMMOnlyActivated(), true);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isMMOnlyActivated(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isMMOnlyActivated(), false);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isMMOnlyActivated(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isMMOnlyActivated(), true);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isMMOnlyActivated(), false);
}

TEST(TestSettings, isRingPolymerMDActivatedTest)
{
    Settings::setJobtype(MM_MD);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);

    Settings::setJobtype(QM_MD);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);

    Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), true);

    Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);

    Settings::setJobtype(MM_OPT);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);

    Settings::setJobtype(NONE);
    EXPECT_EQ(Settings::isRingPolymerMDActivated(), false);
}

TEST(TestSettings, activateKokkosTest)
{
    EXPECT_EQ(Settings::useKokkos(), false);

    Settings::activateKokkos();
    EXPECT_EQ(Settings::useKokkos(), true);
}