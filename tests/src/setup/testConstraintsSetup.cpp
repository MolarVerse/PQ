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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ, Init...

#include "constraintSettings.hpp"   // for getShakeMaxIter, getShakeTolerance, getRattleMaxIter, getRattleTolerance
#include "constraintsSetup.hpp"   // for ConstraintsSetup, setupConstraints
#include "engine.hpp"             // for Engine
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "testSetup.hpp"          // for TestSetup

using namespace setup;

/**
 * @brief tests setupConstraints function for tolerances
 *
 */
TEST_F(TestSetup, setupConstraintTolerances)
{
    settings::ConstraintSettings::setShakeTolerance(1e-6);
    settings::ConstraintSettings::setRattleTolerance(1e-6);

    const auto &constraints = _engine->getConstraints();

    constraints->activateShake();

    ConstraintsSetup constraintsSetup(*_engine);
    constraintsSetup.setup();

    EXPECT_EQ(constraints->getShakeTolerance(), 1e-6);
    EXPECT_EQ(constraints->getRattleTolerance(), 1e-6);
}

/**
 * @brief tests setupConstraints function for max iterations
 *
 */
TEST_F(TestSetup, setupConstraintMaxIter)
{
    settings::ConstraintSettings::setShakeMaxIter(100);
    settings::ConstraintSettings::setRattleMaxIter(100);

    const auto &constraints = _engine->getConstraints();

    constraints->activateShake();

    ConstraintsSetup constraintsSetup(*_engine);
    constraintsSetup.setup();

    EXPECT_EQ(constraints->getShakeMaxIter(), 100);
    EXPECT_EQ(constraints->getRattleMaxIter(), 100);
}

/**
 * @brief tests setupConstraints wrapper function - should not throw
 *
 */
TEST_F(TestSetup, setupConstraints)
{
    settings::ConstraintSettings::setShakeTolerance(999.0);

    const auto &constraints = _engine->getConstraints();

    constraints->deactivateShake();
    EXPECT_NO_THROW(setupConstraints(*_engine));
    const auto shakeToleranceDeactivated = constraints->getShakeTolerance();

    constraints->activateShake();
    EXPECT_NO_THROW(setupConstraints(*_engine));
    const auto shakeToleranceActivated = constraints->getShakeTolerance();

    EXPECT_NE(shakeToleranceDeactivated, shakeToleranceActivated);
}
