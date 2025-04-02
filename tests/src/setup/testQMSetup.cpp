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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS

#include <string>   // for allocator, basic_string

#include "dftbplusRunner.hpp"     // for DFTBPlusRunner
#include "exceptions.hpp"         // for InputFileException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "pyscfRunner.hpp"        // for PySCFRunner
#include "qmRunner.hpp"           // for QMRunner
#include "qmSettings.hpp"         // for QMMethod, QMSettings
#include "qmSetup.hpp"            // for QMSetup, setupQM
#include "qmSetup.hpp"            // for QMSetup
#include "qmmdEngine.hpp"         // for QMMDEngine
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG
#include "turbomoleRunner.hpp"    // for TurbomoleRunner

using setup::QMSetup;
using namespace settings;

TEST(TestQMSetup, setupDftbplus)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("test");
    setupQM.setup();

    EXPECT_EQ(
        typeid(dynamic_cast<QM::DFTBPlusRunner &>(*engine.getQMRunner())),
        typeid(QM::DFTBPlusRunner)
    );

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A qm based jobtype was requested but no external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupPySCF)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::PYSCF);
    settings::QMSettings::setQMScript("test");
    setupQM.setup();

    EXPECT_EQ(
        typeid(dynamic_cast<QM::PySCFRunner &>(*engine.getQMRunner())),
        typeid(QM::PySCFRunner)
    );

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A qm based jobtype was requested but no external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupTurbomoleRunner)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::TURBOMOLE);
    settings::QMSettings::setQMScript("test");
    setupQM.setup();

    EXPECT_EQ(
        typeid(dynamic_cast<QM::TurbomoleRunner &>(*engine.getQMRunner())),
        typeid(QM::TurbomoleRunner)
    );

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A qm based jobtype was requested but no external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupQMFull)
{
    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("test");

    engine::QMMDEngine engine;
    EXPECT_NO_THROW(setup::setupQM(engine));
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderNotSet)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("3ob");
    QMSettings::setIsThirdOrderDftbSet(false);

    qmSetup.setupQMMethodAseDftbPlus();
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetTrue)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("3ob");
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setUseThirdOrderDftb(true);

    qmSetup.setupQMMethodAseDftbPlus();
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetFalse)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("3ob");
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setUseThirdOrderDftb(false);

    qmSetup.setupQMMethodAseDftbPlus();
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusMatsci)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("matsci");
    QMSettings::setIsThirdOrderDftbSet(false);
    QMSettings::setUseThirdOrderDftb(false);

    qmSetup.setupQMMethodAseDftbPlus();
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusCustom)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("custom");
    QMSettings::setIsThirdOrderDftbSet(false);
    QMSettings::setUseThirdOrderDftb(false);

    qmSetup.setupQMMethodAseDftbPlus();
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusHubbardDerivsNo3rdOrder)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::ASEDFTBPLUS);
    QMSettings::setSlakosType("custom");
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setUseThirdOrderDftb(false);
    QMSettings::setIsHubbardDerivsSet(true);
    QMSettings::setHubbardDerivs({{"H", 1.0}});

    ASSERT_THROW_MSG(
        qmSetup.setupQMMethodAseDftbPlus(),
        customException::InputFileException,
        "You have set custom Hubbard derivatives but disabled 3rd order DFTB. "
        "This setup is invalid."
    );
}

TEST(TestQMSetup, setupQMMethodMaceOffInvalidModelSize)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModelType("mace-off");
    QMSettings::setMaceModelSize("medium-omat-0");

    ASSERT_THROW_MSG(
        qmSetup.setupQMMethodMace(),
        customException::InputFileException,
        "The 'medium-omat-0' model size is only compatible with the 'mace_mp' "
        "model type."
    );
}

TEST(TestQMSetup, setupQMMethodMaceRedundantModelPath)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModelType("mace-mp");
    QMSettings::setMaceModelSize("medium-omat-0");
    QMSettings::setMaceModelPath("https://not-a-valid-url");

    ASSERT_THROW_MSG(
        qmSetup.setupQMMethodMace(),
        customException::InputFileException,
        "You have set a custom MACE model path without requesting a custom "
        "mace model size."
        "This setup is invalid."
    );
}

TEST(TestQMSetup, setupQMMethodMaceMissingModelPath)
{
    engine::QMMDEngine engine;
    QMSetup            qmSetup{QMSetup(engine)};

    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceModelType("mace-mp");
    QMSettings::setMaceModelSize("custom");
    QMSettings::setMaceModelPath("");

    ASSERT_THROW_MSG(
        qmSetup.setupQMMethodMace(),
        customException::InputFileException,
        "You have requested a custom MACE model but haven't provided a "
        "MACE model path."
        "This setup is invalid."
    );
}