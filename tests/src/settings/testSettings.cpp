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

#include "generalSettings.hpp"

TEST(TestSettings, stringJobtypeTest)
{
    EXPECT_EQ(JobTypeMeta::toString(JobType::MM_MD), "MM_MD");
    EXPECT_EQ(JobTypeMeta::toString(JobType::QM_MD), "QM_MD");
    EXPECT_EQ(JobTypeMeta::toString(JobType::QMMM_MD), "QMMM_MD");
    EXPECT_EQ(
        JobTypeMeta::toString(JobType::RING_POLYMER_QM_MD),
        "RING_POLYMER_QM_MD"
    );
    EXPECT_EQ(JobTypeMeta::toString(JobType::MM_OPT), "MM_OPT");
}

TEST(TestSettings, setJobtypeTest)
{
    using enum JobType;

    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::getJobtype(), MM_MD);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::getJobtype(), QM_MD);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::getJobtype(), RING_POLYMER_QM_MD);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::getJobtype(), QMMM_MD);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::getJobtype(), MM_OPT);
}

TEST(TestSettings, setFloatingPointTypeTest)
{
    using enum FPType;

    GeneralSettings::setFloatingPointType(FLOAT);
    EXPECT_EQ(GeneralSettings::getFloatingPointType(), FLOAT);

    GeneralSettings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(GeneralSettings::getFloatingPointType(), DOUBLE);

    GeneralSettings::setFloatingPointType(FLOAT);
    EXPECT_EQ(GeneralSettings::getFloatingPointPybindString(), "float32");

    GeneralSettings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(GeneralSettings::getFloatingPointPybindString(), "float64");
}

TEST(TestSettings, setRandomSeedTest)
{
    GeneralSettings::setRandomSeed(73);
    EXPECT_EQ(GeneralSettings::getRandomSeed(), 73);
}

TEST(TestSettings, setIsRandomSeedTest)
{
    GeneralSettings::setIsRandomSeedSet(true);
    EXPECT_EQ(GeneralSettings::isRandomSeedSet(), true);

    GeneralSettings::setIsRandomSeedSet(false);
    EXPECT_EQ(GeneralSettings::isRandomSeedSet(), false);
}

TEST(TestSettings, setDimensionalityTest)
{
    GeneralSettings::setDimensionality(3);
    EXPECT_EQ(GeneralSettings::getDimensionality(), 3);
}

TEST(TestSettings, isQMOnlyJobtypeTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isQMOnlyJobtype(), false);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isQMOnlyJobtype(), true);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isQMOnlyJobtype(), true);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isQMOnlyJobtype(), false);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isQMOnlyJobtype(), false);
}

TEST(TestSettings, isMMOnlyJobtypeTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isMMOnlyJobtype(), true);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isMMOnlyJobtype(), false);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isMMOnlyJobtype(), false);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isMMOnlyJobtype(), false);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isMMOnlyJobtype(), true);
}

TEST(TestSettings, isHybridJobtypeTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isHybridJobtype(), false);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isHybridJobtype(), false);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isHybridJobtype(), false);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isHybridJobtype(), true);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isHybridJobtype(), false);
}

TEST(TestSettings, isMDJobtypeTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isMDJobType(), true);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isMDJobType(), true);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isMDJobType(), true);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isMDJobType(), true);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isMDJobType(), false);
}

TEST(TestSettings, isOptJobtypeTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isOptJobType(), false);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isOptJobType(), false);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isOptJobType(), false);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isOptJobType(), false);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isOptJobType(), true);
}

TEST(TestSettings, isMMActivatedTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isMMActivated(), true);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isMMActivated(), false);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isMMActivated(), false);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isMMActivated(), true);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isMMActivated(), true);
}

TEST(TestSettings, isQMActivatedTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isQMActivated(), false);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isQMActivated(), true);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isQMActivated(), true);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isQMActivated(), true);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isQMActivated(), false);
}

TEST(TestSettings, isRingPolymerMDActivatedTest)
{
    using enum JobType;
    GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(GeneralSettings::isRingPolymerMDActivated(), false);

    GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(GeneralSettings::isRingPolymerMDActivated(), false);

    GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(GeneralSettings::isRingPolymerMDActivated(), true);

    GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(GeneralSettings::isRingPolymerMDActivated(), false);

    GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(GeneralSettings::isRingPolymerMDActivated(), false);
}
