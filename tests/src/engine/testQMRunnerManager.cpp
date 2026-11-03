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
#include "throwWithMessage.hpp"
#include "turbomoleRunner.hpp"

TEST(TestQMRunnerManager, createsTheExternalRunnerForEachMethod)
{
    using engine::QMRunnerManager;

    const auto dftb      = QMRunnerManager::createQMRunner(QMMethod::DFTBPLUS);
    const auto pyscf     = QMRunnerManager::createQMRunner(QMMethod::PYSCF);
    const auto turbomole = QMRunnerManager::createQMRunner(QMMethod::TURBOMOLE);

    EXPECT_NE(dynamic_cast<QM::DFTBPlusRunner *>(dftb.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::PySCFRunner *>(pyscf.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::TurbomoleRunner *>(turbomole.get()), nullptr);
}

TEST(TestQMRunnerManager, eachCallCreatesANewRunner)
{
    const auto first =
        engine::QMRunnerManager::createQMRunner(QMMethod::DFTBPLUS);
    const auto second =
        engine::QMRunnerManager::createQMRunner(QMMethod::DFTBPLUS);

    EXPECT_NE(first.get(), second.get());
}

TEST(TestQMRunnerManager, rejectsMissingQMProgram)
{
    EXPECT_THROW_MSG(
        engine::QMRunnerManager::createQMRunner(QMMethod::NONE),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

#ifndef WITH_ASE

TEST(TestQMRunnerManager, aseMethodsNeedAseAtCompileTime)
{
    using engine::QMRunnerManager;

    EXPECT_THROW_MSG(
        QMRunnerManager::createQMRunner(QMMethod::ASE_DFTBPLUS),
        exc::CompileTimeException,
        "The ASE DFTB+ QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use ASE DFTB+ "
        "type QM methods using: -DBUILD_WITH_ASE=ON"
    );
    EXPECT_THROW_MSG(
        QMRunnerManager::createQMRunner(QMMethod::ASE_XTB),
        exc::CompileTimeException,
        "The ASE xTB QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE xTB "
        "type QM method using: -DBUILD_WITH_ASE=ON"
    );
    EXPECT_THROW_MSG(
        QMRunnerManager::createQMRunner(QMMethod::MACE),
        exc::CompileTimeException,
        "A MACE type QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use MACE type QM "
        "methods using: -DBUILD_WITH_ASE=ON"
    );
    EXPECT_THROW_MSG(
        QMRunnerManager::createQMRunner(QMMethod::FENNOL),
        exc::CompileTimeException,
        "The ASE FeNNol QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE "
        "FeNNol type QM method using: -DBUILD_WITH_ASE=ON"
    );
}

#endif   // WITH_ASE
