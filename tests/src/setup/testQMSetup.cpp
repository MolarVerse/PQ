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
    class DefaultExternalQMRunner : public QM::ExternalQMRunner
    {
       public:
        using QM::ExternalQMRunner::ExternalQMRunner;
        void execute(molsys::SimulationBox & /*simBox*/) override {}
        void writeCoordsFile(molsys::SimulationBox & /*simBox*/) override {}
    };

    void setBuildCompatibleQMScript(QMSettings &qmSettings)
    {
        qmSettings.setQMScript("");
        qmSettings.setQMScriptFullPath("");

        if (std::string_view(SINGULARITY_) == "ON" ||
            std::string_view(STATIC_BUILD_) == "ON")
            qmSettings.setQMScriptFullPath("test");
        else
            qmSettings.setQMScript("test");
    }
}   // namespace

TEST(TestQMSetup, defaultExternalRunnerHooksAreOptional)
{
    QMSettings                 settings;
    DefaultExternalQMRunner    runner{settings};
    molsys::SimulationBox      simBox;
    physicalData::PhysicalData physicalData;
    QM::ExternalQMRunner *volatile baseRunner = &runner;

    EXPECT_NO_THROW(baseRunner->writePointChargeFile(simBox));
    EXPECT_NO_THROW(baseRunner->readStressTensor(simBox.getBox(), physicalData)
    );
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
    Settings           settings;
    engine::QMMDEngine engine(settings);
    auto               setupQM = setup::QMSetup(engine);

    ASSERT_THROW_MSG(
        setupQM.setup(settings.qmSettings),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );

    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    setBuildCompatibleQMScript(settings.qmSettings);
    setupQM.setup(settings.qmSettings);

    test::checkType(engine.getQMRunner(), typeid(QM::DFTBPlusRunner));
}

TEST(TestQMSetup, setupPySCF)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);
    auto               setupQM = setup::QMSetup(engine);

    ASSERT_THROW_MSG(
        setupQM.setup(settings.qmSettings),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );

    settings.qmSettings.setQMMethod(QMMethod::PYSCF);
    setBuildCompatibleQMScript(settings.qmSettings);
    setupQM.setup(settings.qmSettings);

    test::checkType(engine.getQMRunner(), typeid(QM::PySCFRunner));
}

TEST(TestQMSetup, setupTurbomoleRunner)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);
    auto               setupQM = setup::QMSetup(engine);

    ASSERT_THROW_MSG(
        setupQM.setup(settings.qmSettings),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );

    settings.qmSettings.setQMMethod(QMMethod::TURBOMOLE);
    setBuildCompatibleQMScript(settings.qmSettings);
    setupQM.setup(settings.qmSettings);

    test::checkType(engine.getQMRunner(), typeid(QM::TurbomoleRunner));
}

TEST(TestQMSetup, setupQMFull)
{
    Settings settings;
    settings.qmSettings.setQMScript("test");
    engine::QMMDEngine engine(settings);
    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    EXPECT_NO_THROW(setup::setupQM(engine));
}

#ifdef WITH_ASE
TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderNotSet)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings.qmSettings.setSlakosType(SlakosType::THREEOB);
    QMSettings::setIsThirdOrderDftbSet(false);

    setup::QMSetup::setupQMMethodAseDftbPlus(settings.qmSettings);
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetTrue)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings.qmSettings.setSlakosType(SlakosType::THREEOB);
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setUseThirdOrderDftb(true);

    setup::QMSetup::setupQMMethodAseDftbPlus(settings.qmSettings);
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetFalse)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings.qmSettings.setSlakosType(SlakosType::THREEOB);
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus(settings.qmSettings);
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusMatsci)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings.qmSettings.setSlakosType(SlakosType::MATSCI);
    QMSettings::setIsThirdOrderDftbSet(false);
    QMSettings::setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus(settings.qmSettings);
    EXPECT_EQ(QMSettings::useThirdOrderDftb(), false);
}
#endif

TEST(TestQMSetup, setupQMMethodAseDftbPlusCustom)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethod(QMMethod::ASE_DFTBPLUS);
    settings.qmSettings.setSlakosType(SlakosType::CUSTOM);
    settings.qmSettings.setIsThirdOrderDftbSet(false);
    settings.qmSettings.setUseThirdOrderDftb(false);

    setup::QMSetup::setupQMMethodAseDftbPlus(settings.qmSettings);
    EXPECT_EQ(settings.qmSettings.useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMLoopTimeLimitDefault)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup qmSetup(*engine);

    engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    settings.qmSettings.setQMScript("path/To/myQMScript");

    qmSetup.setupWriteInfo(settings.qmSettings);

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
}

TEST(TestQMSetup, setupQMLoopTimeLimitNegative)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup qmSetup(*engine);

    engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    settings.qmSettings.setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(-1.2);

    qmSetup.setupWriteInfo(settings.qmSettings);

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
}

TEST(TestQMSetup, setupQMLoopTimeLimitZero)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup qmSetup(*engine);

    engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    settings.qmSettings.setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(0);

    qmSetup.setupWriteInfo(settings.qmSettings);

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
}

TEST(TestQMSetup, setupQMLoopTimeLimitPositive)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup qmSetup(*engine);

    engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings.qmSettings.setQMMethod(QMMethod::DFTBPLUS);
    settings.qmSettings.setQMScript("path/To/myQMScript");
    QMSettings::setQMLoopTimeLimit(3.14);

    qmSetup.setupWriteInfo(settings.qmSettings);

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
}

TEST(TestQMSetup, setupQMRunnerFennol)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> _engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup _qmSetup(*_engine);

    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    settings.qmSettings.setQMMethod(QMMethod::FENNOL);
    settings.qmSettings.setFennolModelPath("path/To/fennol_model.fnx");
    QMSettings::setUseGPUPreprocessing(false);
    settings::GeneralSettings::setFloatingPointType(FPType::FLOAT);

    _qmSetup.setupWriteInfo(settings.qmSettings);

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
}
