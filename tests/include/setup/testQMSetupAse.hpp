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

#ifndef TESTS__INCLUDE__SETUP__TEST_QM_SETUP_ASE_HPP
#define TESTS__INCLUDE__SETUP__TEST_QM_SETUP_ASE_HPP

#include <gtest/gtest.h>

#include <memory>

#include "qmSettings.hpp"
#include "qmSetup.hpp"
#include "qmmdEngine.hpp"

/**
 * @class TestQMSetupAse
 *
 * @brief test suite for QMSetup ase Runner
 *
 */
class TestQMSetupAse : public ::testing::Test
{
   protected:
    std::unique_ptr<Settings>           _settings;
    std::unique_ptr<engine::QMMDEngine> _engine;
    std::unique_ptr<setup::QMSetup>     _qmSetup;

    void SetUp() override
    {
        _settings = std::make_unique<Settings>();
        _engine   = std::make_unique<engine::QMMDEngine>(*_settings);
        _qmSetup  = std::make_unique<setup::QMSetup>(*_engine);
        _engine->getEngineOutput().getLogOutput().setFilename(
            "default.log",
            false
        );
        _settings->qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    }

    void TearDown() override
    {
        const auto errorCode = std::remove("default.log");
        EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    }
};

#endif   // TESTS__INCLUDE__SETUP__TEST_QM_SETUP_ASE_HPP
