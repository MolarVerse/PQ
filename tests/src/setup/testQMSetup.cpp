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

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include "dftbplusRunner.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "externalQMRunner.hpp"
#include "generalSettings.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "pyscfRunner.hpp"
#include "qmSettings.hpp"
#include "qmSetup.hpp"
#include "qmmdEngine.hpp"
#include "simulationBox.hpp"
#include "testUtils.hpp"
#include "throwWithMessage.hpp"
#include "turbomoleRunner.hpp"

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
        settings::QMSettings::setQMScript("");
        settings::QMSettings::setQMScriptFullPath("");

        if (std::string_view(SINGULARITY_) == "ON" ||
            std::string_view(STATIC_BUILD_) == "ON")
            settings::QMSettings::setQMScriptFullPath("test");
        else
            settings::QMSettings::setQMScript("test");
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

    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::DFTBPlusRunner));

    settings::QMSettings::setQMMethod(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupPySCF)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(QMMethod::PYSCF);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::PySCFRunner));

    settings::QMSettings::setQMMethod(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupTurbomoleRunner)
{
    engine::QMMDEngine engine;
    auto               setupQM = setup::QMSetup(engine);

    settings::QMSettings::setQMMethod(QMMethod::TURBOMOLE);
    setBuildCompatibleQMScript();
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::TurbomoleRunner));

    settings::QMSettings::setQMMethod(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupQMFull)
{
    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("test");

    engine::QMMDEngine engine;
    EXPECT_NO_THROW(setup::setupQM(engine));
}

#ifdef WITH_ASE
TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderNotSet)
{
    engine::QMMDEngine engine;

    settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings::QMSettings::setSlakosType(SlakosType::THREEOB);
    settings::QMSettings::setIsThirdOrderDftbSet(false);

    setup::QMSetup::setupQMMethodAseDftbPlus();
    EXPECT_EQ(settings::QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetTrue)
{
    engine::QMMDEngine engine;

    settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings::QMSettings::setSlakosType(SlakosType::THREEOB);
    settings::QMSettings::setIsThirdOrderDftbSet(true);
    settings::QMSettings::setUseThirdOrderDftb(true);

    setup::QMSetup::setupQMMethodAseDftbPlus();
    EXPECT_EQ(settings::QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetFalse)
{
    engine::QMMDEngine engine;

    settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings::QMSettings::setSlakosType(SlakosType::THREEOB);
    settings::QMSettings::setIsThirdOrderDftbSet(true);
    settings::QMSettings::setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus();
    EXPECT_EQ(settings::QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusMatsci)
{
    engine::QMMDEngine engine;

    settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings::QMSettings::setSlakosType(SlakosType::MATSCI);
    settings::QMSettings::setIsThirdOrderDftbSet(false);
    settings::QMSettings::setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus();
    EXPECT_EQ(settings::QMSettings::useThirdOrderDftb(), false);
}
#endif

TEST(TestQMSetup, setupQMMethodAseDftbPlusCustom)
{
    engine::QMMDEngine engine;

    settings::QMSettings::setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings::QMSettings::setSlakosType(SlakosType::CUSTOM);
    settings::QMSettings::setIsThirdOrderDftbSet(false);
    settings::QMSettings::setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus();
    EXPECT_EQ(settings::QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMLoopTimeLimitDefault)
{
    auto *_engine  = new engine::QMMDEngine();
    auto *_qmSetup = new setup::QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("path/To/myQMScript");

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
    auto *_qmSetup = new setup::QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("path/To/myQMScript");
    settings::QMSettings::setQMLoopTimeLimit(-1.2);

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
    auto *_qmSetup = new setup::QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("path/To/myQMScript");
    settings::QMSettings::setQMLoopTimeLimit(0);

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
    auto *_qmSetup = new setup::QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    settings::QMSettings::setQMScript("path/To/myQMScript");
    settings::QMSettings::setQMLoopTimeLimit(3.14);

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
    auto *_qmSetup = new setup::QMSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings::QMSettings::setQMMethod(QMMethod::FENNOL);
    settings::QMSettings::setFennolModelPath("path/To/fennol_model.fnx");
    settings::QMSettings::setUseGPUPreprocessing(false);
    settings::GeneralSettings::setFloatingPointType(FPType::FLOAT);

    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: FENNOL");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(
        line,
        "         Model path:               path/To/fennol_model.fnx"
    );
    getline(file, line);
    EXPECT_EQ(line, "         Using GPU pre-processing: false");
    getline(file, line);
    EXPECT_EQ(line, "         Using float64:            false");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
    delete _engine;
    delete _qmSetup;
}
