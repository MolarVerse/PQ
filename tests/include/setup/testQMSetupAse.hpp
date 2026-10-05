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

#ifndef _TEST_QMSETUP_ASE_HPP_

#define _TEST_QMSETUP_ASE_HPP_

#include <gtest/gtest.h>

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
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> _engine;
    std::unique_ptr<setup::QMSetup>     _qmSetup;

    void SetUp() override
    {
        _engine  = std::make_unique<engine::QMMDEngine>(settings);
        _qmSetup = std::make_unique<setup::QMSetup>(*_engine);
        _engine->getEngineOutput().getLogOutput().setFilename("default.log");
        settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    }

    void TearDown() override
    {
        const auto errorCode = std::remove("default.log");
        EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
        settings::QMSettings::setQMMethod(QMMethod::NONE);
        settings::QMSettings::setSlakosType(SlakosType::NONE);
        settings::QMSettings::setUseDispersionCorrection(false);
        settings::QMSettings::setUseThirdOrderDftb(false);
        settings::QMSettings::setIsThirdOrderDftbSet(false);
        settings::QMSettings::setHubbardDerivs({});
        settings::QMSettings::setIsHubbardDerivsSet(false);
    }
};

#endif   // _TEST_QMSETUP_ASE_HPP_
