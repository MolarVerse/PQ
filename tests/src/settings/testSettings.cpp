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

    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(generalSettings.getJobtype(), MM_MD);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(generalSettings.getJobtype(), QM_MD);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(generalSettings.getJobtype(), RING_POLYMER_QM_MD);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(generalSettings.getJobtype(), QMMM_MD);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(generalSettings.getJobtype(), MM_OPT);
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

TEST(TestSettings, isQMOnlyJobtypeTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isQMOnlyJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isQMOnlyJobtype(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isQMOnlyJobtype(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isQMOnlyJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isQMOnlyJobtype(generalSettings.getJobtype()), false);
}

TEST(TestSettings, isMMOnlyJobtypeTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isMMOnlyJobtype(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isMMOnlyJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isMMOnlyJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isMMOnlyJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isMMOnlyJobtype(generalSettings.getJobtype()), true);
}

TEST(TestSettings, isHybridJobtypeTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isHybridJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isHybridJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isHybridJobtype(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isHybridJobtype(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isHybridJobtype(generalSettings.getJobtype()), false);
}

TEST(TestSettings, isMDJobtypeTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isMDJobType(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isMDJobType(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isMDJobType(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isMDJobType(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isMDJobType(generalSettings.getJobtype()), false);
}

TEST(TestSettings, isOptJobtypeTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isOptJobType(generalSettings.getJobtype()), true);
}

TEST(TestSettings, isMMActivatedTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isMMActivated(generalSettings.getJobtype()), true);
}

TEST(TestSettings, isQMActivatedTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isQMActivated(generalSettings.getJobtype()), false);
}

TEST(TestSettings, isRingPolymerMDActivatedTest)
{
    using enum JobType;
    GeneralSettings generalSettings;
    generalSettings.setJobtypeByUser(MM_MD);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(QM_MD);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(RING_POLYMER_QM_MD);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), true);

    generalSettings.setJobtypeByUser(QMMM_MD);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), false);

    generalSettings.setJobtypeByUser(MM_OPT);
    EXPECT_EQ(isRingPolymerMDActivated(generalSettings.getJobtype()), false);
}
