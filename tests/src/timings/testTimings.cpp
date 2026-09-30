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

#include "testTimings.hpp"

#include <gtest/gtest.h>

#include <cmath>   // for isfinite

#include "exceptions.hpp"         // for TimerException
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

/**
 * @brief tests that getName returns the constructor-provided name
 *
 */
TEST_F(TestTimingsSection, getName)
{
    EXPECT_EQ(_section->getName(), "test-section");
}

/**
 * @brief tests that a begin/end cycle produces a finite, non-negative
 * elapsed and loop time
 *
 */
TEST_F(TestTimingsSection, beginEndProducesFiniteNonNegativeTimes)
{
    _section->beginTimer();
    _section->endTimer();

    EXPECT_GE(_section->calculateElapsedTime(), 0.0);
    EXPECT_GE(_section->calculateLoopTime(), 0.0);
    EXPECT_TRUE(std::isfinite(_section->calculateElapsedTime()));
    EXPECT_TRUE(std::isfinite(_section->calculateLoopTime()));
}

/**
 * @brief tests that calculateElapsedTime accumulates across multiple
 * begin/end cycles while calculateLoopTime only reflects the most recent
 * one - the cumulative total can therefore never be smaller than the last
 * single step
 *
 */
TEST_F(TestTimingsSection, elapsedTimeAccumulatesAcrossSteps)
{
    _section->beginTimer();
    _section->endTimer();

    _section->beginTimer();
    _section->endTimer();

    EXPECT_GE(_section->calculateElapsedTime(), _section->calculateLoopTime());
}

/**
 * @brief tests that copying a TimingsSection produces an independent deep
 * copy - further timing on the original must not affect the copy's
 * already-recorded elapsed time
 *
 */
TEST_F(TestTimingsSection, copyIsIndependentOfOriginal)
{
    _section->beginTimer();
    _section->endTimer();

    const auto copiedSection     = timings::TimingsSection(*_section);
    const auto copyElapsedAtCopy = copiedSection.calculateElapsedTime();

    _section->beginTimer();
    _section->endTimer();

    EXPECT_EQ(copiedSection.calculateElapsedTime(), copyElapsedAtCopy);
    EXPECT_GE(_section->calculateElapsedTime(), copyElapsedAtCopy);
}

/**
 * @brief tests that a Timer's default name matches its default TimerId
 *
 */
TEST_F(TestTimer, defaultTimerName)
{
    EXPECT_EQ(_timer->getTimerName(), "Default Timings");
}

/**
 * @brief tests that starting then stopping the default-named section
 * succeeds without throwing
 *
 */
TEST_F(TestTimer, startThenStopDoesNotThrow)
{
    EXPECT_NO_THROW(_timer->startTimingsSection());
    EXPECT_NO_THROW(_timer->stopTimingsSection());
}

/**
 * @brief tests that starting and stopping the default-named section twice
 * reuses the same section instead of erroring on the second stop
 *
 */
TEST_F(TestTimer, startStopCycleReusesSameSection)
{
    _timer->startTimingsSection();
    _timer->stopTimingsSection();

    EXPECT_NO_THROW(_timer->startTimingsSection());
    EXPECT_NO_THROW(_timer->stopTimingsSection());

    EXPECT_EQ(_timer->getTimingDetails().size(), 1);
}

/**
 * @brief tests that stopping a timer that was never started throws
 * TimerException instead of silently doing nothing
 *
 */
TEST_F(TestTimer, stopWithoutStartThrows)
{
    EXPECT_THROW_MSG(
        _timer->stopTimingsSection(),
        exc::TimerException,
        "Timer not found"
    );
}

/**
 * @brief tests that requesting an unknown timings section by name throws
 * TimerException
 *
 */
TEST_F(TestTimer, getTimingsSectionRejectsUnknownName)
{
    EXPECT_THROW_MSG(
        (void) _timer->getTimingsSection("unknown-section"),
        exc::TimerException,
        "Timer not found"
    );
}

/**
 * @brief tests that findTimingsSectionIndex returns the sentinel
 * (current size) for an unknown name, and the section's real index once it
 * has been started
 *
 */
TEST_F(TestTimer, findTimingsSectionIndexReturnsSentinelThenRealIndex)
{
    EXPECT_EQ(_timer->findTimingsSectionIndex("Default Timings"), 0);

    _timer->startTimingsSection();

    EXPECT_EQ(_timer->findTimingsSectionIndex("Default Timings"), 0);
    EXPECT_EQ(_timer->findTimingsSectionIndex("still-unknown"), 1);
}
