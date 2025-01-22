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

#include <gtest/gtest.h>   // for Test
#include <stdio.h>         // for remove

#include <memory>   // for allocator

#include "logOutput.hpp"    // for LogOutput
#include "qmSettings.hpp"   // for QMMethod, QMSettings
#include "qmSetup.hpp"      // for QMSetup, setupQM
#include "qmmdEngine.hpp"   // for QMMDEngine

using setup::QMSetup;
using namespace settings;

/**
 * @class TestQMSetupAse
 *
 * @brief test suite for QMSetup ase Runner
 *
 */
class TestQMSetupAse : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _engine  = new engine::QMMDEngine();
        _qmSetup = new QMSetup(*_engine);
        _engine->getEngineOutput().getLogOutput().setFilename("default.log");
        QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    }

    void TearDown() override
    {
        delete _engine;
        delete _qmSetup;
        ::remove("default.log");
        QMSettings::setQMMethod(QMMethod::NONE);
        QMSettings::setSlakosType("none");
        QMSettings::setUseDispersionCorrection(false);
        QMSettings::setUseThirdOrderDftb(false);
        QMSettings::setIsThirdOrderDftbSet(false);
        QMSettings::setHubbardDerivs({});
        QMSettings::setIsHubbardDerivsSet(false);
    }

    engine::QMMDEngine *_engine;
    QMSetup            *_qmSetup;
};

#endif   // _TEST_QMSETUP_ASE_HPP_