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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include "atom.hpp"
#include "dftbplusRunner.hpp"
#include "exceptions.hpp"
#include "externalQMRunner.hpp"
#include "fileSettings.hpp"
#include "physicalData.hpp"
#include "pyscfRunner.hpp"
#include "qmSettings.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "stringUtilities.hpp"
#include "throwWithMessage.hpp"
#include "turbomoleRunner.hpp"

namespace
{
    void writeFile(const std::string_view fileName, const std::string_view text)
    {
        auto file = std::ofstream(std::string(fileName));
        file << text;
    }

    class ExternalQMRunnerHarness : public QM::ExternalQMRunner
    {
       private:
        bool _sawStaleResults = false;

       public:
        void writeCoordsFile(molsys::SimulationBox & /*simBox*/) override {}

        void execute(molsys::SimulationBox & /*simBox*/) override
        {
            _sawStaleResults =
                std::filesystem::exists(
                    settings::FileSettings::getQMForcesTempFileName()
                ) ||
                std::filesystem::exists(
                    settings::FileSettings::getQMChargesTempFileName()
                ) ||
                std::filesystem::exists(
                    settings::FileSettings::getStressTensorTempFileName()
                );

            writeFile(
                settings::FileSettings::getQMForcesTempFileName(),
                "0\n0 0 0\n"
            );
            writeFile(
                settings::FileSettings::getQMChargesTempFileName(),
                "0\n"
            );
        }

        void runCommand(
            const std::string_view command,
            const std::string_view program
        ) const
        {
            _executeCommand(command, program);
        }

        [[nodiscard]] bool sawStaleResults() const { return _sawStaleResults; }
    };

    template <class Runner>
    class CommandCaptureRunner : public Runner
    {
       private:
        mutable std::string _command;

       protected:
        void _executeCommand(
            const std::string_view command,
            const std::string_view /*program*/
        ) const override
        {
            _command = command;
        }

       public:
        [[nodiscard]] const std::string &getCommand() const { return _command; }
    };

}   // namespace

class ExternalQMRunnerTest : public testing::Test
{
   protected:
    std::filesystem::path      _originalPath;
    std::filesystem::path      _workPath;
    molsys::SimulationBox      _simulationBox;
    physicalData::PhysicalData _physicalData;
    ExternalQMRunnerHarness    _runner;
    QM::DFTBPlusRunner         _dftbRunner;

    settings::QMMethod _qmMethod;
    settings::JobType  _jobType;
    bool               _removeNetForce;
    double             _timeLimit;
    std::string        _qmScript;
    std::string        _dftbFile;

    static void readForceFile(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        QM::ExternalQMRunner::_readForceFile(simulationBox, physicalData);
    }

    static void readChargeFile(molsys::SimulationBox &simulationBox)
    {
        QM::ExternalQMRunner::_readChargeFile(simulationBox);
    }

    void SetUp() override
    {
        _qmMethod       = settings::QMSettings::getQMMethod();
        _jobType        = settings::Settings::getJobtype();
        _removeNetForce = settings::QMSettings::getRemoveNetForce();
        _timeLimit      = settings::QMSettings::getQMLoopTimeLimit();
        _qmScript       = settings::QMSettings::getQMScript();
        _dftbFile       = settings::FileSettings::getDFTBFileName();
        _originalPath   = std::filesystem::current_path();

        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        _workPath = std::filesystem::temp_directory_path() /
                    ("pq external qm; " + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directory(_workPath));
        std::filesystem::current_path(_workPath);

        settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
        settings::QMSettings::setRemoveNetForce(false);
        settings::QMSettings::setQMLoopTimeLimit(0.0);
        settings::Settings::setJobtype(settings::JobType::QM_MD);

        auto atom = std::make_shared<molsys::Atom>();
        atom->setName("H");
        _simulationBox.addAtom(atom);
        _simulationBox.setBoxDimensions({10.0, 10.0, 10.0});
    }

    void TearDown() override
    {
        settings::QMSettings::setQMMethod(_qmMethod);
        settings::QMSettings::setRemoveNetForce(_removeNetForce);
        settings::QMSettings::setQMLoopTimeLimit(_timeLimit);
        settings::QMSettings::setQMScript(_qmScript);
        settings::FileSettings::setDFTBFileName(_dftbFile);
        settings::Settings::setJobtype(_jobType);

        std::filesystem::current_path(_originalPath);
        std::error_code error;
        std::filesystem::remove_all(_workPath, error);
        EXPECT_FALSE(error);
    }

    std::filesystem::path configureQuotedScript(
        QM::ExternalQMRunner &runner
    ) const
    {
        const auto scriptDirectory =
            _workPath / "working path; $(touch qm-injected)";
        std::filesystem::create_directory(scriptDirectory);

        const auto *const scriptName = "runner's script; touch qm-injected; #";

        const auto scriptFile = scriptDirectory / scriptName;
        writeFile(scriptFile.string(), "");

        runner.setScriptPath(scriptDirectory.string() + '/');
        settings::QMSettings::setQMScript(scriptName);

        return scriptFile;
    }
};

TEST_F(ExternalQMRunnerTest, propagatesCommandFailure)
{
#if defined(_WIN32)
    try
    {
        _runner.runCommand("true", "External QM");
        FAIL() << "Expected command execution to be rejected on Windows";
    }
    catch (const exc::QMRunnerException &error)
    {
        EXPECT_THAT(error.what(), HasSubstr("not supported on Windows"));
    }
#else
    EXPECT_NO_THROW(_runner.runCommand("true", "External QM"));

    try
    {
        _runner.runCommand("false", "External QM");
        FAIL() << "Expected the failed command to throw";
    }
    catch (const exc::QMRunnerException &error)
    {
        EXPECT_THAT(
            error.what(),
            testing::HasSubstr("External QM command failed")
        );
    }
#endif
}

TEST_F(ExternalQMRunnerTest, quotesDftbCommandArguments)
{
    auto       runner    = CommandCaptureRunner<QM::DFTBPlusRunner>();
    const auto path      = configureQuotedScript(runner);
    const auto inputFile = std::string("input file; touch qm-injected");
    settings::FileSettings::setDFTBFileName(inputFile);

    runner.execute(_simulationBox);

    EXPECT_EQ(
        std::format(
            "{} 0 0 0 {} {}",
            utilities::shellQuote(path.string()),
            utilities::shellQuote(inputFile),
            utilities::shellQuote(
                settings::FileSettings::getPointChargeFileName()
            )
        ),
        runner.getCommand()
    );
    EXPECT_FALSE(std::filesystem::exists(_workPath / "qm-injected"));
}

TEST_F(ExternalQMRunnerTest, quotesPyscfCommandArguments)
{
    auto       runner = CommandCaptureRunner<QM::PySCFRunner>();
    const auto path   = configureQuotedScript(runner);

    runner.execute(_simulationBox);

    EXPECT_EQ(
        std::format(
            "python {} > {}",
            utilities::shellQuote(path.string()),
            utilities::shellQuote("pyscf.out")
        ),
        runner.getCommand()
    );
    EXPECT_FALSE(std::filesystem::exists(_workPath / "qm-injected"));
}

TEST_F(ExternalQMRunnerTest, quotesTurbomoleCommandArguments)
{
    auto       runner = CommandCaptureRunner<QM::TurbomoleRunner>();
    const auto path   = configureQuotedScript(runner);

    runner.execute(_simulationBox);

    EXPECT_EQ(
        std::format(
            "{} 0 1 0 {} {}",
            utilities::shellQuote(path.string()),
            utilities::shellQuote(settings::FileSettings::getTMFileName()),
            utilities::shellQuote(
                settings::FileSettings::getPointChargeFileName()
            )
        ),
        runner.getCommand()
    );
    EXPECT_FALSE(std::filesystem::exists(_workPath / "qm-injected"));
}

TEST_F(ExternalQMRunnerTest, removesStaleResultsBeforeExecution)
{
    writeFile(settings::FileSettings::getQMForcesTempFileName(), "stale");
    writeFile(settings::FileSettings::getQMChargesTempFileName(), "stale");
    writeFile(settings::FileSettings::getStressTensorTempFileName(), "stale");

    EXPECT_NO_THROW(_runner.run(
        _simulationBox,
        _physicalData,
        molsys::Periodicity::NON_PERIODIC
    ));
    EXPECT_FALSE(_runner.sawStaleResults());
    EXPECT_FALSE(
        std::filesystem::exists(
            settings::FileSettings::getStressTensorTempFileName()
        )
    );
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteForces)
{
    writeFile(settings::FileSettings::getQMForcesTempFileName(), "0\n0 0\n");

    EXPECT_THROW_MSG(
        readForceFile(_simulationBox, _physicalData),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS force file \"qm_forces\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteForces)
{
    writeFile(
        settings::FileSettings::getQMForcesTempFileName(),
        "0\nnan 0 0\n"
    );

    EXPECT_THROW_MSG(
        readForceFile(_simulationBox, _physicalData),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS force file \"qm_forces\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteCharges)
{
    auto atom = std::make_shared<molsys::Atom>();
    atom->setName("H");
    _simulationBox.addAtom(atom);

    writeFile(settings::FileSettings::getQMChargesTempFileName(), "0\n");

    EXPECT_THROW_MSG(
        readChargeFile(_simulationBox),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS charge file \"qm_charges\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteCharges)
{
    writeFile(settings::FileSettings::getQMChargesTempFileName(), "nan\n");

    EXPECT_THROW_MSG(
        readChargeFile(_simulationBox),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS charge file \"qm_charges\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteStressTensor)
{
    writeFile(
        settings::FileSettings::getStressTensorTempFileName(),
        "0 0 0\n0 0 0\n"
    );

    EXPECT_THROW_MSG(
        _dftbRunner.readStressTensor(_simulationBox.getBox(), _physicalData),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS stress tensor \"stress_tensor\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteStressTensor)
{
    writeFile(
        settings::FileSettings::getStressTensorTempFileName(),
        "nan 0 0\n0 0 0\n0 0 0\n"
    );

    EXPECT_THROW_MSG(
        _dftbRunner.readStressTensor(_simulationBox.getBox(), _physicalData),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS stress tensor \"stress_tensor\""
    );
}
