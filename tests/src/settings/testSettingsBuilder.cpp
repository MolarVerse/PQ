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
#include "defaults.hpp"
#include "settings.hpp"
#include "testSettings.hpp"

TEST(TestSettingsBuilder, OverridesAreAppliedAndTheRestKeepsItsDefault)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setShakeMaxIter, 100)
            .with(&ConstraintSettings::setShakeTolerance, 1.0e-8)
            .build();

    EXPECT_EQ(100U, settings.getShakeMaxIter());
    EXPECT_DOUBLE_EQ(1.0e-8, settings.getShakeTolerance());
    EXPECT_EQ(defaults::RATTLE_MAX_ITER_DEFAULT, settings.getRattleMaxIter());
    EXPECT_DOUBLE_EQ(
        defaults::RATTLE_TOLERANCE_DEFAULT,
        settings.getRattleTolerance()
    );
    EXPECT_FALSE(settings.isShakeActivated());
}

TEST(TestSettingsBuilder, ApplyReachesEverythingThatIsNotAPlainSetter)
{
    const auto settings = testSettings::make<ConstraintSettings>()
                              .apply([](auto &s) { s.activateShake(); })
                              .apply([](auto &s) { s.activateMShake(); })
                              .build();

    EXPECT_TRUE(settings.isShakeActivated());
    EXPECT_TRUE(settings.isMShakeActivated());
    EXPECT_FALSE(settings.isDistanceConstraintsActivated());
}

TEST(TestSettingsBuilder, LaterOverridesWin)
{
    const auto settings = testSettings::make<ConstraintSettings>()
                              .with(&ConstraintSettings::setShakeMaxIter, 10)
                              .with(&ConstraintSettings::setShakeMaxIter, 20)
                              .build();

    EXPECT_EQ(20U, settings.getShakeMaxIter());
}

TEST(TestSettingsBuilder, BuildReturnsACopyAndTheBuilderCanBeReused)
{
    auto builder = testSettings::make<ConstraintSettings>().with(
        &ConstraintSettings::setShakeMaxIter,
        10
    );
    const auto first = builder.build();
    builder.with(&ConstraintSettings::setShakeMaxIter, 99);
    const auto second = builder.build();

    EXPECT_EQ(10U, first.getShakeMaxIter());
    EXPECT_EQ(99U, second.getShakeMaxIter());
}

TEST(TestSettingsBuilder, TheRootSettingsAreBuiltTheSameWay)
{
    const auto settings = testSettings::make<Settings>()
                              .apply(
                                  [](auto &s)
                                  {
                                      s.constraintSettings.setShakeMaxIter(7);
                                      s.constraintSettings.activateShake();
                                  }
                              )
                              .build();

    EXPECT_EQ(7U, settings.constraintSettings.getShakeMaxIter());
    EXPECT_TRUE(settings.constraintSettings.isShakeActivated());
}

TEST(TestSettingsBuilder, TwoBuildersDoNotShareState)
{
    const auto a = testSettings::make<ConstraintSettings>()
                       .with(&ConstraintSettings::setShakeMaxIter, 1)
                       .build();
    const auto b = testSettings::make<ConstraintSettings>().build();

    EXPECT_EQ(1U, a.getShakeMaxIter());
    EXPECT_EQ(defaults::SHAKE_MAX_ITER_DEFAULT, b.getShakeMaxIter());
}
