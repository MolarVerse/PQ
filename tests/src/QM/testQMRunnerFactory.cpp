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

#include "external/dftbplusRunner.hpp"
#include "external/pyscfRunner.hpp"
#include "external/turbomoleRunner.hpp"
#include "qmRunnerFactory.hpp"
#include "qmSettings.hpp"
#include "throwWithMessage.hpp"

TEST(TestQMRunnerManager, createsTheExternalRunnerForEachMethod)
{
    QMSettings settings;
    settings.setQMMethod(QMMethod::DFTBPLUS);
    const auto dftb = QM::QMRunnerFactory::createQMRunner(settings);
    settings.setQMMethod(QMMethod::PYSCF);
    const auto pyscf = QM::QMRunnerFactory::createQMRunner(settings);
    settings.setQMMethod(QMMethod::TURBOMOLE);
    const auto turbomole = QM::QMRunnerFactory::createQMRunner(settings);

    EXPECT_NE(dynamic_cast<QM::DFTBPlusRunner *>(dftb.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::PySCFRunner *>(pyscf.get()), nullptr);
    EXPECT_NE(dynamic_cast<QM::TurbomoleRunner *>(turbomole.get()), nullptr);
}

TEST(TestQMRunnerManager, eachCallCreatesANewRunner)
{
    QMSettings settings;
    settings.setQMMethod(QMMethod::DFTBPLUS);
    const auto first  = QM::QMRunnerFactory::createQMRunner(settings);
    const auto second = QM::QMRunnerFactory::createQMRunner(settings);

    EXPECT_NE(first.get(), second.get());
}

TEST(TestQMRunnerManager, rejectsMissingQMProgram)
{
    QMSettings settings;
    EXPECT_THROW_MSG(
        const auto _ = QM::QMRunnerFactory::createQMRunner(settings),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

#ifndef WITH_ASE

TEST(TestQMRunnerManager, aseMethodsNeedAseAtCompileTime)
{
    QMSettings settings;
    settings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    EXPECT_THROW_MSG(
        QM::QMRunnerFactory::createQMRunner(settings),
        exc::CompileTimeException,
        "The ASE DFTB+ QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use ASE DFTB+ "
        "type QM methods using: -DBUILD_WITH_ASE=ON"
    );
    settings.setQMMethod(QMMethod::ASE_XTB);
    EXPECT_THROW_MSG(
        QM::QMRunnerFactory::createQMRunner(settings),
        exc::CompileTimeException,
        "The ASE xTB QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE xTB "
        "type QM method using: -DBUILD_WITH_ASE=ON"
    );
    settings.setQMMethod(QMMethod::MACE);
    EXPECT_THROW_MSG(
        QM::QMRunnerFactory::createQMRunner(settings),
        exc::CompileTimeException,
        "A MACE type QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use MACE type QM "
        "methods using: -DBUILD_WITH_ASE=ON"
    );
    settings.setQMMethod(QMMethod::FENNOL);
    EXPECT_THROW_MSG(
        QM::QMRunnerFactory::createQMRunner(settings),
        exc::CompileTimeException,
        "The ASE FeNNol QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE "
        "FeNNol type QM method using: -DBUILD_WITH_ASE=ON"
    );
}

#endif   // WITH_ASE
