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

#include <memory>

#include "dftbplusRunner.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "pyscfRunner.hpp"
#include "qmRunnerManager.hpp"
#include "qmSettings.hpp"
#include "settings.hpp"
#include "throwWithMessage.hpp"
#include "turbomoleRunner.hpp"

TEST(TestQMRunnerManager, createsTheExternalRunnerForEachMethod)
{
    using engine::QMRunnerManager;

    Settings settings;
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    const auto dftb = QMRunnerManager::createQMRunner(
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    settings.qmSettings.setQMMethodByUser(QMMethod::PYSCF);
    const auto pyscf = QMRunnerManager::createQMRunner(
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    settings.qmSettings.setQMMethodByUser(QMMethod::TURBOMOLE);
    const auto turbomole = QMRunnerManager::createQMRunner(
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    EXPECT_NE(dynamic_cast<QM::DFTBPlusRunner *>(dftb.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::PySCFRunner *>(pyscf.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::TurbomoleRunner *>(turbomole.get()), nullptr);
}

TEST(TestQMRunnerManager, eachCallCreatesANewRunner)
{
    Settings settings;
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    const auto first = engine::QMRunnerManager::createQMRunner(
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );
    const auto second = engine::QMRunnerManager::createQMRunner(
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    EXPECT_NE(first.get(), second.get());
}

TEST(TestQMRunnerManager, rejectsMissingQMProgram)
{
    Settings settings;
    settings.qmSettings.setQMMethodByUser(QMMethod::NONE);
    EXPECT_THROW_MSG(
        auto _ = engine::QMRunnerManager::createQMRunner(
            settings.qmSettings,
            settings.maceSettings,
            settings.fennolSettings,
            settings.dftbSettings,
            settings.xtbSettings,
            settings.externalQMSettings
        ),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

#ifndef WITH_ASE

TEST(TestQMRunnerManager, aseMethodsNeedAseAtCompileTime)
{
    using engine::QMRunnerManager;
    Settings settings;

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    EXPECT_THROW_MSG(
        const auto _ = QMRunnerManager::createQMRunner(
            settings.qmSettings,
            settings.maceSettings,
            settings.fennolSettings,
            settings.dftbSettings,
            settings.xtbSettings,
            settings.externalQMSettings
        ),
        exc::CompileTimeException,
        "The ASE DFTB+ QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use ASE DFTB+ "
        "type QM methods using: -DBUILD_WITH_ASE=ON"
    );
    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_XTB);
    EXPECT_THROW_MSG(
        const auto _ = QMRunnerManager::createQMRunner(
            settings.qmSettings,
            settings.maceSettings,
            settings.fennolSettings,
            settings.dftbSettings,
            settings.xtbSettings,
            settings.externalQMSettings
        ),
        exc::CompileTimeException,
        "The ASE xTB QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE xTB "
        "type QM method using: -DBUILD_WITH_ASE=ON"
    );
    settings.qmSettings.setQMMethodByUser(QMMethod::MACE);
    EXPECT_THROW_MSG(
        const auto _ = QMRunnerManager::createQMRunner(
            settings.qmSettings,
            settings.maceSettings,
            settings.fennolSettings,
            settings.dftbSettings,
            settings.xtbSettings,
            settings.externalQMSettings
        ),
        exc::CompileTimeException,
        "A MACE type QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use MACE type QM "
        "methods using: -DBUILD_WITH_ASE=ON"
    );
    settings.qmSettings.setQMMethodByUser(QMMethod::FENNOL);
    EXPECT_THROW_MSG(
        const auto _ = QMRunnerManager::createQMRunner(
            settings.qmSettings,
            settings.maceSettings,
            settings.fennolSettings,
            settings.dftbSettings,
            settings.xtbSettings,
            settings.externalQMSettings
        ),
        exc::CompileTimeException,
        "The ASE FeNNol QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE "
        "FeNNol type QM method using: -DBUILD_WITH_ASE=ON"
    );
}

#endif   // WITH_ASE
