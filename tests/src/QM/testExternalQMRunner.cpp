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
#include "turbomoleRunner.hpp"

using customException::QMRunnerException;
using molsys::Atom;
using molsys::Periodicity;
using molsys::SimulationBox;
using physicalData::PhysicalData;
using settings::FileSettings;
using settings::JobType;
using settings::QMMethod;
using settings::QMSettings;
using settings::Settings;
using testing::HasSubstr;

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
        void writeCoordsFile(SimulationBox & /*simBox*/) override {}

        void execute(SimulationBox & /*simBox*/) override
        {
            _sawStaleResults = std::filesystem::exists(
                                   FileSettings::getQMForcesTempFileName()
                               ) ||
                               std::filesystem::exists(
                                   FileSettings::getQMChargesTempFileName()
                               ) ||
                               std::filesystem::exists(
                                   FileSettings::getStressTensorTempFileName()
                               );

            writeFile(FileSettings::getQMForcesTempFileName(), "0\n0 0 0\n");
            writeFile(FileSettings::getQMChargesTempFileName(), "0\n");
        }

        void runCommand(
            const std::string_view command,
            const std::string_view program
        ) const
        {
            executeCommand(command, program);
        }

        [[nodiscard]] bool sawStaleResults() const { return _sawStaleResults; }
    };

    template <class Runner>
    class CommandCaptureRunner : public Runner
    {
       private:
        mutable std::string _command;

       protected:
        void executeCommand(
            const std::string_view command,
            const std::string_view /*program*/
        ) const override
        {
            _command = command;
        }

       public:
        [[nodiscard]] const std::string &getCommand() const { return _command; }
    };

    class ExternalQMRunnerTest : public testing::Test
    {
       protected:
        std::filesystem::path   _originalPath;
        std::filesystem::path   _workPath;
        SimulationBox           _simulationBox;
        PhysicalData            _physicalData;
        ExternalQMRunnerHarness _runner;
        QM::DFTBPlusRunner      _dftbRunner;

        QMMethod    _qmMethod;
        JobType     _jobType;
        bool        _removeNetForce;
        double      _timeLimit;
        std::string _qmScript;
        std::string _dftbFile;

        void SetUp() override
        {
            _qmMethod       = QMSettings::getQMMethod();
            _jobType        = Settings::getJobtype();
            _removeNetForce = QMSettings::getRemoveNetForce();
            _timeLimit      = QMSettings::getQMLoopTimeLimit();
            _qmScript       = QMSettings::getQMScript();
            _dftbFile       = FileSettings::getDFTBFileName();
            _originalPath   = std::filesystem::current_path();

            const auto stamp =
                std::chrono::steady_clock::now().time_since_epoch().count();
            _workPath = std::filesystem::temp_directory_path() /
                        ("pq external qm; " + std::to_string(stamp));
            ASSERT_TRUE(std::filesystem::create_directory(_workPath));
            std::filesystem::current_path(_workPath);

            QMSettings::setQMMethod(QMMethod::DFTBPLUS);
            QMSettings::setRemoveNetForce(false);
            QMSettings::setQMLoopTimeLimit(0.0);
            Settings::setJobtype(JobType::QM_MD);

            auto atom = std::make_shared<Atom>();
            atom->setName("H");
            _simulationBox.addAtom(atom);
            _simulationBox.setBoxDimensions({10.0, 10.0, 10.0});
        }

        void TearDown() override
        {
            QMSettings::setQMMethod(_qmMethod);
            QMSettings::setRemoveNetForce(_removeNetForce);
            QMSettings::setQMLoopTimeLimit(_timeLimit);
            QMSettings::setQMScript(_qmScript);
            FileSettings::setDFTBFileName(_dftbFile);
            Settings::setJobtype(_jobType);

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

            const auto *const scriptName =
                "runner's script; touch qm-injected; #";

            const auto scriptFile = scriptDirectory / scriptName;
            writeFile(scriptFile.string(), "");

            runner.setScriptPath(scriptDirectory.string() + '/');
            QMSettings::setQMScript(scriptName);

            return scriptFile;
        }
    };
}   // namespace

TEST_F(ExternalQMRunnerTest, propagatesCommandFailure)
{
#if defined(_WIN32)
    try
    {
        _runner.runCommand("true", "External QM");
        FAIL() << "Expected command execution to be rejected on Windows";
    }
    catch (const QMRunnerException &error)
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
    catch (const QMRunnerException &error)
    {
        EXPECT_THAT(error.what(), HasSubstr("External QM command failed"));
    }
#endif
}

TEST_F(ExternalQMRunnerTest, quotesDftbCommandArguments)
{
    auto       runner    = CommandCaptureRunner<QM::DFTBPlusRunner>();
    const auto path      = configureQuotedScript(runner);
    const auto inputFile = std::string("input file; touch qm-injected");
    FileSettings::setDFTBFileName(inputFile);

    runner.execute(_simulationBox);

    EXPECT_EQ(
        std::format(
            "{} 0 0 0 {} {}",
            utilities::shellQuote(path.string()),
            utilities::shellQuote(inputFile),
            utilities::shellQuote(FileSettings::getPointChargeFileName())
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
            utilities::shellQuote(FileSettings::getTMFileName()),
            utilities::shellQuote(FileSettings::getPointChargeFileName())
        ),
        runner.getCommand()
    );
    EXPECT_FALSE(std::filesystem::exists(_workPath / "qm-injected"));
}

TEST_F(ExternalQMRunnerTest, removesStaleResultsBeforeExecution)
{
    writeFile(FileSettings::getQMForcesTempFileName(), "stale");
    writeFile(FileSettings::getQMChargesTempFileName(), "stale");
    writeFile(FileSettings::getStressTensorTempFileName(), "stale");

    EXPECT_NO_THROW(
        _runner.run(_simulationBox, _physicalData, Periodicity::NON_PERIODIC)
    );
    EXPECT_FALSE(_runner.sawStaleResults());
    EXPECT_FALSE(
        std::filesystem::exists(FileSettings::getStressTensorTempFileName())
    );
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteForces)
{
    writeFile(FileSettings::getQMForcesTempFileName(), "0\n0 0\n");

    EXPECT_THROW(
        _runner.readForceFile(_simulationBox, _physicalData),
        QMRunnerException
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteForces)
{
    writeFile(FileSettings::getQMForcesTempFileName(), "0\nnan 0 0\n");

    EXPECT_THROW(
        _runner.readForceFile(_simulationBox, _physicalData),
        QMRunnerException
    );
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteCharges)
{
    auto atom = std::make_shared<Atom>();
    atom->setName("H");
    _simulationBox.addAtom(atom);

    writeFile(FileSettings::getQMChargesTempFileName(), "0\n");

    EXPECT_THROW(_runner.readChargeFile(_simulationBox), QMRunnerException);
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteCharges)
{
    writeFile(FileSettings::getQMChargesTempFileName(), "nan\n");

    EXPECT_THROW(_runner.readChargeFile(_simulationBox), QMRunnerException);
}

TEST_F(ExternalQMRunnerTest, rejectsIncompleteStressTensor)
{
    writeFile(FileSettings::getStressTensorTempFileName(), "0 0 0\n0 0 0\n");

    EXPECT_THROW(
        _dftbRunner.readStressTensor(_simulationBox.getBox(), _physicalData),
        QMRunnerException
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteStressTensor)
{
    writeFile(
        FileSettings::getStressTensorTempFileName(),
        "nan 0 0\n0 0 0\n0 0 0\n"
    );

    EXPECT_THROW(
        _dftbRunner.readStressTensor(_simulationBox.getBox(), _physicalData),
        QMRunnerException
    );
}
