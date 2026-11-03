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

#include "constraintSettings.hpp"
#include "referenceSettings.hpp"
#include "waterModelSettings.hpp"

TEST(ConstraintSettingsTest, ShakeMaxIterRoundTrip)
{
    ConstraintSettings settings;
    settings.setShakeMaxIter(500U);
    EXPECT_EQ(settings.getShakeMaxIter(), 500U);

    settings.setShakeMaxIter(1U);
    EXPECT_EQ(settings.getShakeMaxIter(), 1U);
}

TEST(ConstraintSettingsTest, RattleMaxIterRoundTrip)
{
    ConstraintSettings settings;
    settings.setRattleMaxIter(750U);
    EXPECT_EQ(settings.getRattleMaxIter(), 750U);
}

TEST(ConstraintSettingsTest, MShakeMaxIterRoundTrip)
{
    ConstraintSettings settings;
    settings.setMShakeMaxIter(5U);
    EXPECT_EQ(settings.getMShakeMaxIter(), 5U);

    settings.setMShakeMaxIter(2U);
    EXPECT_EQ(settings.getMShakeMaxIter(), 2U);
}

TEST(ConstraintSettingsTest, ShakeToleranceRoundTrip)
{
    ConstraintSettings settings;
    settings.setShakeTolerance(1.0e-4);
    EXPECT_DOUBLE_EQ(settings.getShakeTolerance(), 1.0e-4);

    settings.setShakeTolerance(1.0e-12);
    EXPECT_DOUBLE_EQ(settings.getShakeTolerance(), 1.0e-12);
}

TEST(ConstraintSettingsTest, RattleToleranceRoundTrip)
{
    ConstraintSettings settings;
    settings.setRattleTolerance(1.0e-6);
    EXPECT_DOUBLE_EQ(settings.getRattleTolerance(), 1.0e-6);
}

TEST(ConstraintSettingsTest, MShakeToleranceRoundTrip)
{
    ConstraintSettings settings;
    settings.setMShakeTolerance(3.0e-5);
    EXPECT_DOUBLE_EQ(settings.getMShakeTolerance(), 3.0e-5);

    settings.setMShakeTolerance(1.1e-10);
    EXPECT_DOUBLE_EQ(settings.getMShakeTolerance(), 1.1e-10);
}

/**
 * @brief finalize() must activate shake when the configured water model
 * has a rigid geometry (e.g. SPC), even if shake was never explicitly
 * requested.
 */
TEST(ConstraintSettingsTest, FinalizeActivatesShakeForRigidWaterModel)
{
    ConstraintSettings settings;
    ASSERT_FALSE(settings.isShakeActivated());

    WaterModelSettings waterModelSettings;
    waterModelSettings.setWaterIntraModel(WaterIntraModel::SPC);
    ASSERT_TRUE(waterModelSettings.getRigidWaterGeometry().has_value());

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings, waterModelSettings);

    EXPECT_TRUE(settings.isShakeActivated());
    EXPECT_TRUE(referenceSettings.getReferenceFileNames().contains(
        ReferenceFiles::RATTLE_FILE
    ));
}

/**
 * @brief finalize() must not activate shake for a water model without a
 * rigid geometry (e.g. SPC_FW), nor add the rattle reference.
 */
TEST(ConstraintSettingsTest, FinalizeDoesNotActivateShakeForFlexibleWaterModel)
{
    ConstraintSettings settings;

    WaterModelSettings waterModelSettings;
    waterModelSettings.setWaterIntraModel(WaterIntraModel::SPC_FW);
    ASSERT_FALSE(waterModelSettings.getRigidWaterGeometry().has_value());

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings, waterModelSettings);

    EXPECT_FALSE(settings.isShakeActivated());
    EXPECT_TRUE(referenceSettings.getReferenceFileNames().empty());
}

/**
 * @brief finalize() must add the rattle reference whenever shake ends up
 * active, regardless of whether that came from an explicit
 * activateShake() call or from a rigid water model.
 */
TEST(ConstraintSettingsTest, FinalizeAddsRattleReferenceForExplicitShake)
{
    ConstraintSettings settings;
    settings.activateShake();

    WaterModelSettings waterModelSettings;

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings, waterModelSettings);

    EXPECT_TRUE(settings.isShakeActivated());
    EXPECT_TRUE(referenceSettings.getReferenceFileNames().contains(
        ReferenceFiles::RATTLE_FILE
    ));
}
