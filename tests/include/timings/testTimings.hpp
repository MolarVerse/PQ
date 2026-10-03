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

#ifndef _TEST_TIMINGS_HPP_

#define _TEST_TIMINGS_HPP_

#include <gtest/gtest.h>

#include "timer.hpp"
#include "timingsSection.hpp"

/**
 * @class TestTimingsSection
 *
 * @brief Fixture for TimingsSection tests.
 *
 */
class TestTimingsSection : public ::testing::Test
{
   protected:
    std::unique_ptr<timings::TimingsSection> _section;

    void SetUp() override
    {
        _section = std::make_unique<timings::TimingsSection>("test-section");
    }
};

/**
 * @class TestTimer
 *
 * @brief Fixture for Timer tests.
 *
 */
class TestTimer : public ::testing::Test
{
   protected:
    std::unique_ptr<timings::Timer> _timer;

    void SetUp() override { _timer = std::make_unique<timings::Timer>(); }
};

#endif   // _TEST_TIMINGS_HPP_
