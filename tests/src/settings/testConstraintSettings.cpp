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
#include "gtest/gtest.h"

TEST(ConstraintSettingsTest, ShakeMaxIterRoundTrip)
{
    settings::ConstraintSettings::setShakeMaxIter(500u);
    EXPECT_EQ(settings::ConstraintSettings::getShakeMaxIter(), 500u);

    settings::ConstraintSettings::setShakeMaxIter(1u);
    EXPECT_EQ(settings::ConstraintSettings::getShakeMaxIter(), 1u);
}

TEST(ConstraintSettingsTest, RattleMaxIterRoundTrip)
{
    settings::ConstraintSettings::setRattleMaxIter(750u);
    EXPECT_EQ(settings::ConstraintSettings::getRattleMaxIter(), 750u);
}

TEST(ConstraintSettingsTest, MShakeMaxIterRoundTrip)
{
    settings::ConstraintSettings::setMShakeMaxIter(5u);
    EXPECT_EQ(settings::ConstraintSettings::getMShakeMaxIter(), 5u);

    settings::ConstraintSettings::setMShakeMaxIter(2u);
    EXPECT_EQ(settings::ConstraintSettings::getMShakeMaxIter(), 2u);
}

TEST(ConstraintSettingsTest, ShakeToleranceRoundTrip)
{
    settings::ConstraintSettings::setShakeTolerance(1.0e-4);
    EXPECT_DOUBLE_EQ(
        settings::ConstraintSettings::getShakeTolerance(),
        1.0e-4
    );

    settings::ConstraintSettings::setShakeTolerance(1.0e-12);
    EXPECT_DOUBLE_EQ(
        settings::ConstraintSettings::getShakeTolerance(),
        1.0e-12
    );
}

TEST(ConstraintSettingsTest, RattleToleranceRoundTrip)
{
    settings::ConstraintSettings::setRattleTolerance(1.0e-6);
    EXPECT_DOUBLE_EQ(
        settings::ConstraintSettings::getRattleTolerance(),
        1.0e-6
    );
}

TEST(ConstraintSettingsTest, MShakeToleranceRoundTrip)
{
    settings::ConstraintSettings::setMShakeTolerance(3.0e-5);
    EXPECT_DOUBLE_EQ(
        settings::ConstraintSettings::getMShakeTolerance(),
        3.0e-5
    );

    settings::ConstraintSettings::setMShakeTolerance(1.1e-10);
    EXPECT_DOUBLE_EQ(
        settings::ConstraintSettings::getMShakeTolerance(),
        1.1e-10
    );
}

