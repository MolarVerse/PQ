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

// #include <memory>   // for allocator

#include "settings.hpp"
// #include "exceptions.hpp"         // for UserInputException
// #include gtest.h"          // for Message, TestPartResult
// #include "qmSettings.hpp"         // for QMSettings, QMMethod
// #include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

TEST(TestSettings, stringJobtypeTest)
{
    EXPECT_EQ(string(settings::JobType::MM_MD), "MM_MD");
    EXPECT_EQ(string(settings::JobType::QM_MD), "QM_MD");
    EXPECT_EQ(string(settings::JobType::QMMM_MD), "QMMM_MD");
    EXPECT_EQ(
        string(settings::JobType::RING_POLYMER_QM_MD),
        "RING_POLYMER_QM_MD"
    );
    EXPECT_EQ(string(settings::JobType::MM_OPT), "MM_OPT");
    EXPECT_EQ(string(settings::JobType::NONE), "NONE");
}

TEST(TestSettings, setJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype("MmMD");
    EXPECT_EQ(settings::Settings::getJobtype(), MM_MD);

    settings::Settings::setJobtype("qMMd");
    EXPECT_EQ(settings::Settings::getJobtype(), QM_MD);

    settings::Settings::setJobtype("RinG-POLymer_QMMd");
    EXPECT_EQ(settings::Settings::getJobtype(), RING_POLYMER_QM_MD);

    settings::Settings::setJobtype("qMmmMd");
    EXPECT_EQ(settings::Settings::getJobtype(), QMMM_MD);

    settings::Settings::setJobtype("MMoPT");
    EXPECT_EQ(settings::Settings::getJobtype(), MM_OPT);

    settings::Settings::setJobtype("not-a-jobtype");
    EXPECT_EQ(settings::Settings::getJobtype(), NONE);

    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::getJobtype(), MM_MD);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::getJobtype(), QM_MD);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::getJobtype(), RING_POLYMER_QM_MD);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::getJobtype(), QMMM_MD);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::getJobtype(), MM_OPT);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::getJobtype(), NONE);
}

TEST(TestSettings, setFloatingPointTypeTest)
{
    using enum settings::FPType;
    settings::Settings::setFloatingPointType("FlOAt");
    EXPECT_EQ(settings::Settings::getFloatingPointType(), FLOAT);

    settings::Settings::setFloatingPointType("DOUble");
    EXPECT_EQ(settings::Settings::getFloatingPointType(), DOUBLE);

    settings::Settings::setFloatingPointType("not-a-floating-point-type");
    EXPECT_EQ(settings::Settings::getFloatingPointType(), DOUBLE);

    settings::Settings::setFloatingPointType(FLOAT);
    EXPECT_EQ(settings::Settings::getFloatingPointType(), FLOAT);

    settings::Settings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(settings::Settings::getFloatingPointType(), DOUBLE);

    settings::Settings::setFloatingPointType(FLOAT);
    EXPECT_EQ(settings::Settings::getFloatingPointPybindString(), "float32");

    settings::Settings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(settings::Settings::getFloatingPointPybindString(), "float64");
}

TEST(TestSettings, setRandomSeedTest)
{
    settings::Settings::setRandomSeed(73);
    EXPECT_EQ(settings::Settings::getRandomSeed(), 73);
}

TEST(TestSettings, setIsRandomSeedTest)
{
    settings::Settings::setIsRandomSeedSet(true);
    EXPECT_EQ(settings::Settings::isRandomSeedSet(), true);

    settings::Settings::setIsRandomSeedSet(false);
    EXPECT_EQ(settings::Settings::isRandomSeedSet(), false);
}

TEST(TestSettings, setIsRingPolymerMDActivatedTest)
{
    settings::Settings::setIsRingPolymerMDActivated(true);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), true);

    settings::Settings::setIsRingPolymerMDActivated(false);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);
}

TEST(TestSettings, setDimensionalityTest)
{
    settings::Settings::setDimensionality(3);
    EXPECT_EQ(settings::Settings::getDimensionality(), 3);
}

TEST(TestSettings, isQMOnlyJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), true);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), true);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isQMOnlyJobtype(), false);
}

TEST(TestSettings, isMMOnlyJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), true);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), false);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isMMOnlyJobtype(), false);
}

TEST(TestSettings, isHybridJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), false);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), true);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isHybridJobtype(), false);
}

TEST(TestSettings, isMDJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isMDJobType(), true);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isMDJobType(), true);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isMDJobType(), true);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isMDJobType(), true);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isMDJobType(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isMDJobType(), false);
}

TEST(TestSettings, isOptJobtypeTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isOptJobType(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isOptJobType(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isOptJobType(), false);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isOptJobType(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isOptJobType(), true);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isOptJobType(), false);
}

TEST(TestSettings, isMMActivatedTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isMMActivated(), true);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isMMActivated(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isMMActivated(), false);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isMMActivated(), true);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isMMActivated(), true);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isMMActivated(), false);
}

TEST(TestSettings, isQMActivatedTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isQMActivated(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isQMActivated(), true);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isQMActivated(), true);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isQMActivated(), true);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isQMActivated(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isQMActivated(), false);
}

TEST(TestSettings, isQMOnlyActivatedTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), true);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), true);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isQMOnlyActivated(), false);
}

TEST(TestSettings, isMMOnlyActivatedTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), true);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), false);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), true);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isMMOnlyActivated(), false);
}

TEST(TestSettings, isRingPolymerMDActivatedTest)
{
    using enum settings::JobType;
    settings::Settings::setJobtype(MM_MD);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);

    settings::Settings::setJobtype(QM_MD);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);

    settings::Settings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), true);

    settings::Settings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);

    settings::Settings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);

    settings::Settings::setJobtype(NONE);
    EXPECT_EQ(settings::Settings::isRingPolymerMDActivated(), false);
}
