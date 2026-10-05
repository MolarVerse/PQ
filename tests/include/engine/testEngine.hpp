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

#ifndef _TEST_ENGINE_HPP_

#define _TEST_ENGINE_HPP_

#include <gtest/gtest.h>

#include "mmmdEngine.hpp"
#include "timingsSettings.hpp"

/**
 * @class TestEngine
 *
 * @brief Fixture for base Engine behavior, exercised through the concrete
 * MMMDEngine (an MM MD engine, used as a plain default-constructible
 * stand-in the same way other test suites already do).
 *
 */
class TestEngine : public ::testing::Test
{
   protected:
    Settings                            _settings;
    std::unique_ptr<engine::MMMDEngine> _engine;

    void SetUp() override
    {
        _engine = std::make_unique<engine::MMMDEngine>(_settings);
    }

    void TearDown() override
    {
        settings::TimingsSettings::setStepCount(0);
        settings::TimingsSettings::setTimeStep(0.0);
    }
};

#endif   // _TEST_ENGINE_HPP_
