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

#include <cstdlib>
#include <filesystem>
#include <string>        // for allocator, basic_string
#include <string_view>   // for string_view

#include "dftbplusRunner.hpp"   // for DFTBPlusRunner
#include "exceptions.hpp"       // for InputFileException
#include "externalQMRunner.hpp"
#include "gtest/gtest.h"   // for Message, TestPartResult
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "pyscfRunner.hpp"   // for PySCFRunner
#include "qmSettings.hpp"    // for QMMethod, QMSettings
#include "qmSetup.hpp"       // for QMSetup, setupQM
#include "qmSetup.hpp"       // for QMSetup
#include "qmmdEngine.hpp"    // for QMMDEngine
#include "settings.hpp"      // for Settings
#include "simulationBox.hpp"
#include "testUtils.hpp"
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG
#include "turbomoleRunner.hpp"    // for TurbomoleRunner

using setup::QMSetup;
using namespace settings;

namespace
{
    class DefaultExternalQMRunner final : public QM::ExternalQMRunner
    {
       public:
        void execute(molsys::SimulationBox & /*simBox*/) override {}
        void writeCoordsFile(molsys::SimulationBox & /*simBox*/) override {}
    };

    void setBuildCompatibleQMScript()
    {
        QMSettings::setQMScript("");
        QMSettings::setQMScriptFullPath("");

        if (std::string_view(SINGULARITY_) == "ON" ||
            std::string_view(STATIC_BUILD_) == "ON")
            QMSettings::setQMScriptFullPath("test");
        else
            QMSettings::setQMScript("test");
    }
}   // namespace

TEST(TestQMSetup, defaultExternalRunnerHooksAreOptional)
{
    DefaultExternalQMRunner    runner;
    molsys::SimulationBox      simBox;
    molsys::OrthorhombicBox    box;
    physicalData::PhysicalData physicalData;
    QM::ExternalQMRunner *volatile baseRunner = &runner;

    EXPECT_NO_THROW(baseRunner->writePointChargeFile(simBox));
    EXPECT_NO_THROW(baseRunner->readStressTensor(box, physicalData));
}

TEST(TestQMSetup, resolvesBundledQMScript)
{
    const auto script = QM::bundledQMScriptPath("pyscf_hf.py");

    EXPECT_EQ(std::filesystem::path(script).filename(), "pyscf_hf.py");
    EXPECT_TRUE(std::filesystem::is_regular_file(script));

    if (const auto *expected = std::getenv("PQ_TEST_EXPECTED_SCRIPT_DIR"))
    {
        EXPECT_EQ(
            std::filesystem::path(script).parent_path(),
            std::filesystem::path(expected)
        );
    }
}

TEST(TestQMSetup, setupDftbplus)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::DFTBPlusRunner));

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupPySCF)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::PYSCF);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::PySCFRunner));

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupTurbomoleRunner)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(settings::QMMethod::TURBOMOLE);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::TurbomoleRunner));

    settings::QMSettings::setQMMethod(settings::QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        customException::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
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

#ifdef WITH_ASE
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
#endif

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

TEST(TestQMSetup, setupQMLoopTimeLimitDefault)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    QMSettings::setQMScript("path/To/myQMScript");

    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: DFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM script: path/To/myQMScript");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM looptime limit: 3600 s");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}

TEST(TestQMSetup, setupQMLoopTimeLimitNegative)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    QMSettings::setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(-1.2);

    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: DFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM script: path/To/myQMScript");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM looptime limit: unlimited");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}

TEST(TestQMSetup, setupQMLoopTimeLimitZero)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    QMSettings::setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(0);

    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: DFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM script: path/To/myQMScript");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM looptime limit: unlimited");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}

TEST(TestQMSetup, setupQMLoopTimeLimitPositive)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    QMSettings::setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(3.14);

    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: DFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM script: path/To/myQMScript");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         QM looptime limit: 3.14 s");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}

TEST(TestQMSetup, setupQMRunnerFennol)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    QMSettings::setQMMethod(QMMethod::FENNOL);
    QMSettings::setFennolModelPath("path/To/fennol_model.fnx");
    QMSettings::setUseGPUPreprocessing(false);
    Settings::setFloatingPointType(FPType::FLOAT);

    _qmSetup->setupWriteInfo();

    // clang-format off
    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: FeNNol");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         Model path:               path/To/fennol_model.fnx");
    getline(file, line);
    EXPECT_EQ(line, "         Using GPU pre-processing: false");
    getline(file, line);
    EXPECT_EQ(line, "         Using float64:            false");
    // clang-format on

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}
