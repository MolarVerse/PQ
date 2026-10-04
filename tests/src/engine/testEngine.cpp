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

#include "testEngine.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests calculateTotalSimulationTime with a fresh engine (step 1,
 * no restart offset)
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeFreshRun)
{
    settings::TimingsSettings::setStepCount(0);
    settings::TimingsSettings::setTimeStep(0.5);

    // a freshly constructed engine starts at step 1
    EXPECT_EQ(_engine->getStep(), 1);
    EXPECT_DOUBLE_EQ(_engine->calculateTotalSimulationTime(), 0.5);
}

/**
 * @brief tests that calculateTotalSimulationTime folds in the restart step
 * offset (settings::TimingsSettings::getStepCount()), not just the
 * engine's own step counter
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeIncludesRestartOffset)
{
    settings::TimingsSettings::setStepCount(1000);
    settings::TimingsSettings::setTimeStep(0.5);

    // effStep = engine step (1) + restart offset (1000) = 1001
    EXPECT_DOUBLE_EQ(_engine->calculateTotalSimulationTime(), 1001 * 0.5);
}

/**
 * @brief tests calculateTotalSimulationTime scales linearly with the time
 * step
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeScalesWithTimeStep)
{
    settings::TimingsSettings::setStepCount(0);
    settings::TimingsSettings::setTimeStep(2.0);

    EXPECT_DOUBLE_EQ(_engine->calculateTotalSimulationTime(), 2.0);
}

/**
 * @brief tests isForceFieldNonCoulombicsActivated / isGuffActivated
 * reflect the underlying force field's non-Coulombic flag, and are always
 * each other's inverse
 *
 */
TEST_F(TestEngine, isForceFieldNonCoulombicsActivatedTracksForceField)
{
    EXPECT_FALSE(_engine->isForceFieldNonCoulombicsActivated());
    EXPECT_TRUE(_engine->isGuffActivated());

    _engine->getForceField()->activateNonCoulombic();

    EXPECT_TRUE(_engine->isForceFieldNonCoulombicsActivated());
    EXPECT_FALSE(_engine->isGuffActivated());

    _engine->getForceField()->deactivateNonCoulombic();

    EXPECT_FALSE(_engine->isForceFieldNonCoulombicsActivated());
    EXPECT_TRUE(_engine->isGuffActivated());
}
