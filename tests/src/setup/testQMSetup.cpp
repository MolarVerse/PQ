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
#include "units.hpp"

namespace
{
    class DefaultExternalQMRunner : public QM::ExternalQMRunner
    {
       public:
        explicit DefaultExternalQMRunner(
            const QMSettings         &qmSettings,
            const ExternalQMSettings &externalQMSettings
        )
            : QM::ExternalQMRunner(qmSettings, externalQMSettings)
        {
        }

        void execute(
            molsys::SimulationBox & /*simBox*/,
            const FileSettings & /*fileSettings*/
        ) override
        {
        }

        // the optional hooks are protected -> expose the base defaults
        void callWritePointChargeFile(molsys::SimulationBox &simBox)
        {
            _writePointChargeFile(simBox, "point_charges");
        }

        void callReadStressTensor(
            molsys::Box                &box,
            physicalData::PhysicalData &physData
        )
        {
            _readStressTensor(box, physData, "stress_tensor");
        }

       protected:
        void _writeCoordsFile(molsys::SimulationBox & /*simBox*/) override {}
    };

    // the full path variant is valid for all build types (including
    // singularity and static builds)
    void setBuildCompatibleQMScript(ExternalQMSettings &externalQMSettings)
    {
        externalQMSettings.setQMScriptFullPathByUser("test");
        externalQMSettings.finalize();
    }

}   // namespace

TEST(TestQMSetup, defaultExternalRunnerHooksAreOptional)
{
    QMSettings                 qmSettings;
    ExternalQMSettings         externalQMSettings;
    DefaultExternalQMRunner    runner{qmSettings, externalQMSettings};
    molsys::SimulationBox      simBox;
    physicalData::PhysicalData physicalData;

    EXPECT_NO_THROW(runner.callWritePointChargeFile(simBox));
    EXPECT_NO_THROW(runner.callReadStressTensor(simBox.getBox(), physicalData));
}

TEST(TestQMSetup, resolvesBundledQMScript)
{
    ExternalQMSettings externalQMSettings;
    externalQMSettings.setQMScriptByUser("pyscf_hf.py");
    externalQMSettings.finalize();
    const auto script = externalQMSettings.getQMScript();

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
    auto               setupQM = setup::QMSetup(
        engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    setBuildCompatibleQMScript(settings.externalQMSettings);
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::DFTBPlusRunner));

    settings.qmSettings.setQMMethodByUser(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupPySCF)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);
    auto               setupQM = setup::QMSetup(
        engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    settings.qmSettings.setQMMethodByUser(QMMethod::PYSCF);
    setBuildCompatibleQMScript(settings.externalQMSettings);
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::PySCFRunner));

    settings.qmSettings.setQMMethodByUser(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupTurbomoleRunner)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);
    auto               setupQM = setup::QMSetup(
        engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    settings.qmSettings.setQMMethodByUser(QMMethod::TURBOMOLE);
    setBuildCompatibleQMScript(settings.externalQMSettings);
    setupQM.setup();

    test::checkType(*engine.getQMRunner(), typeid(QM::TurbomoleRunner));

    settings.qmSettings.setQMMethodByUser(QMMethod::NONE);

    ASSERT_THROW_MSG(
        setupQM.setup(),
        exc::InputFileException,
        "A QM based jobtype was requested but no valid external program via "
        "\"qm_prog\" provided"
    );
}

TEST(TestQMSetup, setupQMFull)
{
    Settings settings;
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    setBuildCompatibleQMScript(settings.externalQMSettings);

    engine::QMMDEngine engine(settings);
    EXPECT_NO_THROW(setup::setupQM(engine));
}

#ifdef WITH_ASE
TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderNotSet)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    settings.dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB);
    settings.finalize();

    EXPECT_EQ(settings.dftbSettings.useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetTrue)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    settings.dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB);
    settings.dftbSettings.setUseThirdOrderDftbByUser(true);
    settings.finalize();

    EXPECT_EQ(settings.dftbSettings.useThirdOrderDftb(), true);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlus3ob3rdOrderSetFalse)
{
    // this test has nothing to do with the current setup approach
    // it tests now basically the functionality of settings
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    settings.dftbSettings.setSlakosTypeByUser(SlakosType::THREEOB);
    settings.dftbSettings.setUseThirdOrderDftbByUser(false);
    settings.finalize();

    EXPECT_EQ(settings.dftbSettings.useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMMethodAseDftbPlusMatsci)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    settings.dftbSettings.setSlakosTypeByUser(SlakosType::MATSCI);
    settings.dftbSettings.setUseThirdOrderDftbByUser(false);
    settings.finalize();

    EXPECT_EQ(settings.dftbSettings.useThirdOrderDftb(), false);
}
#endif

TEST(TestQMSetup, setupQMMethodAseDftbPlusCustom)
{
    Settings           settings;
    engine::QMMDEngine engine(settings);

    settings.qmSettings.setQMMethodByUser(QMMethod::ASE_DFTBPLUS);
    settings.dftbSettings.setSlakosTypeByUser(SlakosType::CUSTOM);
    settings.dftbSettings.setUseThirdOrderDftbByUser(false);
    settings.finalize();

    EXPECT_EQ(settings.dftbSettings.useThirdOrderDftb(), false);
}

TEST(TestQMSetup, setupQMLoopTimeLimitDefault)
{
    Settings                            settings;
    std::unique_ptr<engine::QMMDEngine> engine =
        std::make_unique<engine::QMMDEngine>(settings);
    setup::QMSetup qmSetup(
        *engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    settings.externalQMSettings.setQMScriptFullPathByUser("path/To/myQMScript");
    settings.externalQMSettings.finalize();

    qmSetup.setupWriteInfo();

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
    setup::QMSetup qmSetup(
        *engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    settings.externalQMSettings.setQMScriptFullPathByUser("path/To/myQMScript");
    settings.externalQMSettings.finalize();
    settings.qmSettings.setQMLoopTimeLimitByUser(-1.2 * s);

    qmSetup.setupWriteInfo();

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
    setup::QMSetup qmSetup(
        *engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    settings.externalQMSettings.setQMScriptFullPathByUser("path/To/myQMScript");
    settings.externalQMSettings.finalize();
    settings.qmSettings.setQMLoopTimeLimitByUser(0 * s);

    qmSetup.setupWriteInfo();

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
    setup::QMSetup qmSetup(
        *engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
    settings.externalQMSettings.setQMScriptFullPathByUser("path/To/myQMScript");
    settings.externalQMSettings.finalize();
    settings.qmSettings.setQMLoopTimeLimitByUser(3.14 * s);

    qmSetup.setupWriteInfo();

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
    setup::QMSetup _qmSetup(
        *_engine,
        settings.qmSettings,
        settings.maceSettings,
        settings.fennolSettings,
        settings.dftbSettings,
        settings.xtbSettings,
        settings.externalQMSettings
    );

    _engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    settings.qmSettings.setQMMethodByUser(QMMethod::FENNOL);
    settings.fennolSettings.setFennolModelPathByUser(
        "path/To/fennol_model.fnx"
    );
    settings.fennolSettings.setUseGPUPreprocessingByUser(false);
    settings::GeneralSettings::setFloatingPointType(FPType::FLOAT);

    _qmSetup.setupWriteInfo();

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
