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

#include "units.hpp"

/**
 * @brief tests calculateTotalSimulationTime with a fresh engine (step 1,
 * no restart offset)
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeFreshRun)
{
    TimingsSettings::setStepCount(0);
    const auto timeStep = 0.5 * fs;

    // a freshly constructed engine starts at step 1
    EXPECT_EQ(_engine->getStep(), 1);
    EXPECT_EQ(_engine->calculateTotalSimulationTime(timeStep), timeStep);
}

/**
 * @brief tests that calculateTotalSimulationTime folds in the restart step
 * offset (TimingsSettings::getStepCount()), not just the
 * engine's own step counter
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeIncludesRestartOffset)
{
    TimingsSettings::setStepCount(1000);
    const auto timeStep = 0.5 * fs;

    // effStep = engine step (1) + restart offset (1000) = 1001
    EXPECT_EQ(_engine->calculateTotalSimulationTime(timeStep), 1001 * timeStep);
}

/**
 * @brief tests calculateTotalSimulationTime scales linearly with the time
 * step
 *
 */
TEST_F(TestEngine, calculateTotalSimulationTimeScalesWithTimeStep)
{
    TimingsSettings::setStepCount(0);
    const auto timeStep = 2.0 * fs;

    EXPECT_EQ(_engine->calculateTotalSimulationTime(timeStep), timeStep);
}
