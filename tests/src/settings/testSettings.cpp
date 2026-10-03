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
    EXPECT_EQ(JobTypeMeta::toString(JobType::NONE), "NONE");
}

TEST(TestSettings, setJobtypeTest)
{
    using enum JobType;

    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), MM_MD);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), QM_MD);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), RING_POLYMER_QM_MD);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), QMMM_MD);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), MM_OPT);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::getJobtype(), NONE);
}

TEST(TestSettings, setFloatingPointTypeTest)
{
    using enum FPType;

    settings::GeneralSettings::setFloatingPointType(FLOAT);
    EXPECT_EQ(settings::GeneralSettings::getFloatingPointType(), FLOAT);

    settings::GeneralSettings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(settings::GeneralSettings::getFloatingPointType(), DOUBLE);

    settings::GeneralSettings::setFloatingPointType(FLOAT);
    EXPECT_EQ(
        settings::GeneralSettings::getFloatingPointPybindString(),
        "float32"
    );

    settings::GeneralSettings::setFloatingPointType(DOUBLE);
    EXPECT_EQ(
        settings::GeneralSettings::getFloatingPointPybindString(),
        "float64"
    );
}

TEST(TestSettings, setRandomSeedTest)
{
    settings::GeneralSettings::setRandomSeed(73);
    EXPECT_EQ(settings::GeneralSettings::getRandomSeed(), 73);
}

TEST(TestSettings, setIsRandomSeedTest)
{
    settings::GeneralSettings::setIsRandomSeedSet(true);
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), true);

    settings::GeneralSettings::setIsRandomSeedSet(false);
    EXPECT_EQ(settings::GeneralSettings::isRandomSeedSet(), false);
}

TEST(TestSettings, setIsRingPolymerMDActivatedTest)
{
    settings::GeneralSettings::setIsRingPolymerMDActivated(true);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), true);

    settings::GeneralSettings::setIsRingPolymerMDActivated(false);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);
}

TEST(TestSettings, setDimensionalityTest)
{
    settings::GeneralSettings::setDimensionality(3);
    EXPECT_EQ(settings::GeneralSettings::getDimensionality(), 3);
}

TEST(TestSettings, isQMOnlyJobtypeTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), true);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), true);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyJobtype(), false);
}

TEST(TestSettings, isMMOnlyJobtypeTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), true);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyJobtype(), false);
}

TEST(TestSettings, isHybridJobtypeTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), false);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), true);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isHybridJobtype(), false);
}

TEST(TestSettings, isMDJobtypeTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), true);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), true);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), true);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), true);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isMDJobType(), false);
}

TEST(TestSettings, isOptJobtypeTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), false);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), true);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isOptJobType(), false);
}

TEST(TestSettings, isMMActivatedTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), false);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), true);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isMMActivated(), false);
}

TEST(TestSettings, isQMActivatedTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), true);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), true);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), true);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isQMActivated(), false);
}

TEST(TestSettings, isQMOnlyActivatedTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), true);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), true);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isQMOnlyActivated(), false);
}

TEST(TestSettings, isMMOnlyActivatedTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), true);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), true);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isMMOnlyActivated(), false);
}

TEST(TestSettings, isRingPolymerMDActivatedTest)
{
    using enum JobType;
    settings::GeneralSettings::setJobtype(MM_MD);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);

    settings::GeneralSettings::setJobtype(QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);

    settings::GeneralSettings::setJobtype(RING_POLYMER_QM_MD);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), true);

    settings::GeneralSettings::setJobtype(QMMM_MD);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);

    settings::GeneralSettings::setJobtype(MM_OPT);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);

    settings::GeneralSettings::setJobtype(NONE);
    EXPECT_EQ(settings::GeneralSettings::isRingPolymerMDActivated(), false);
}
