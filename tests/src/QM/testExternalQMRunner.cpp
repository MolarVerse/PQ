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
#include <format>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "atom.hpp"
#include "constants.hpp"
#include "dftbplusRunner.hpp"
#include "exceptions.hpp"
#include "externalQMRunner.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "pyscfRunner.hpp"
#include "qmSettings.hpp"
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

    std::string readFile(const std::string_view fileName)
    {
        auto file   = std::ifstream(std::string(fileName));
        auto buffer = std::stringstream();
        buffer << file.rdbuf();
        return buffer.str();
    }

    class PeriodicDftbRunner : public QM::DFTBPlusRunner
    {
       public:
        PeriodicDftbRunner() { _periodicity = molsys::Periodicity::XYZ; }
    };

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

       public:
        [[nodiscard]]
        const std::string &getCommand() const
        {
            return _command;
        }

       protected:
        void _executeCommand(
            const std::string_view command,
            const std::string_view /*program*/
        ) const override
        {
            _command = command;
        }
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

    QMMethod    _qmMethod;
    JobType     _jobType;
    bool        _removeNetForce;
    double      _timeLimit;
    std::string _qmScript;
    std::string _dftbFile;

    static void _readForceFile(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        QM::ExternalQMRunner::_readForceFile(simulationBox, physicalData);
    }

    static void _readChargeFile(molsys::SimulationBox &simulationBox)
    {
        QM::ExternalQMRunner::_readChargeFile(simulationBox);
    }

    void SetUp() override
    {
        _qmMethod       = settings::QMSettings::getQMMethod();
        _jobType        = settings::GeneralSettings::getJobtype();
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

        settings::QMSettings::setQMMethod(QMMethod::DFTBPLUS);
        settings::QMSettings::setRemoveNetForce(false);
        settings::QMSettings::setQMLoopTimeLimit(0.0);
        settings::GeneralSettings::setJobtype(JobType::QM_MD);

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
        settings::GeneralSettings::setJobtype(_jobType);

        std::filesystem::current_path(_originalPath);
        std::error_code error;
        std::filesystem::remove_all(_workPath, error);
        EXPECT_FALSE(error);
    }

    std::filesystem::path _configureQuotedScript(
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
    const auto path      = _configureQuotedScript(runner);
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
    const auto path   = _configureQuotedScript(runner);

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
    const auto path   = _configureQuotedScript(runner);

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
        _readForceFile(_simulationBox, _physicalData),
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
        _readForceFile(_simulationBox, _physicalData),
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
        _readChargeFile(_simulationBox),
        exc::QMRunnerException,
        "Incomplete DFTBPLUS charge file \"qm_charges\""
    );
}

TEST_F(ExternalQMRunnerTest, rejectsNonFiniteCharges)
{
    writeFile(settings::FileSettings::getQMChargesTempFileName(), "nan\n");

    EXPECT_THROW_MSG(
        _readChargeFile(_simulationBox),
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

TEST_F(ExternalQMRunnerTest, rejectsUniaxialPeriodicity)
{
    EXPECT_THROW_MSG(
        _runner.run(_simulationBox, _physicalData, molsys::Periodicity::X),
        exc::QMRunnerException,
        "External QM runners only available for non- and 3D-periodic "
        "calculations."
    );
}

TEST_F(ExternalQMRunnerTest, rejectsBiaxialPeriodicity)
{
    EXPECT_THROW_MSG(
        _runner.run(_simulationBox, _physicalData, molsys::Periodicity::XY),
        exc::QMRunnerException,
        "External QM runners only available for non- and 3D-periodic "
        "calculations."
    );
}

TEST_F(ExternalQMRunnerTest, dftbExecuteRejectsMissingScript)
{
    const auto scriptDirectory = (_workPath / "no-such-directory").string();
    _dftbRunner.setScriptPath(scriptDirectory + '/');
    settings::QMSettings::setQMScript("this-script-does-not-exist.sh");

    EXPECT_THROW_MSG(
        _dftbRunner.execute(_simulationBox),
        exc::InputFileException,
        std::format(
            "DFTB+ script file \"{}/this-script-does-not-exist.sh\" does "
            "not exist.",
            scriptDirectory
        )
    );
}

TEST_F(ExternalQMRunnerTest, pyscfExecuteRejectsMissingScript)
{
    const auto scriptDirectory = (_workPath / "no-such-directory").string();

    auto runner = QM::PySCFRunner();
    runner.setScriptPath(scriptDirectory + '/');
    settings::QMSettings::setQMScript("this-script-does-not-exist.py");

    EXPECT_THROW_MSG(
        runner.execute(_simulationBox),
        exc::InputFileException,
        std::format(
            "PySCF script file \"{}/this-script-does-not-exist.py\" does "
            "not exist.",
            scriptDirectory
        )
    );
}

TEST_F(ExternalQMRunnerTest, turbomoleExecuteRejectsMissingScript)
{
    const auto scriptDirectory = (_workPath / "no-such-directory").string();

    auto runner = QM::TurbomoleRunner();
    runner.setScriptPath(scriptDirectory + '/');
    settings::QMSettings::setQMScript("this-script-does-not-exist.sh");

    EXPECT_THROW_MSG(
        runner.execute(_simulationBox),
        exc::InputFileException,
        std::format(
            "Turbomole script file \"{}/this-script-does-not-exist.sh\" "
            "does not exist.",
            scriptDirectory
        )
    );
}

namespace
{
    const auto NON_PERIODIC_COORDS = std::string(
        "3  C\n"
        "H  O  \n"
        "    1     1\t  0.000000000000\t  0.000000000000\t  0.000000000000\n"
        "    2     2\t  1.500000000000\t  2.250000000000\t -3.000000000000\n"
        "    3     1\t  0.500000000000\t  0.500000000000\t  0.500000000000\n"
    );

    const auto CELL_LINES = std::string(
        "           \t  0.000000000000\t  0.000000000000\t  0.000000000000\n"
        "           \t 10.000000000000\t  0.000000000000\t  0.000000000000\n"
        "           \t  0.000000000000\t 12.000000000000\t  0.000000000000\n"
        "           \t  0.000000000000\t  0.000000000000\t 14.000000000000\n"
    );

    const auto POINT_CHARGES = std::string(
        "  1.000000000000\t  2.000000000000\t  3.000000000000\t  "
        "0.500000000000\n"
        "  4.000000000000\t  5.000000000000\t  6.000000000000\t "
        "-0.800000000000\n"
    );

    std::string expectedTurbomolePointCharges()
    {
        const auto line = [](double x, double y, double z, double charge)
        {
            return std::format(
                "{:16.12f}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                x * ANGSTROM_TO_BOHR,
                y * ANGSTROM_TO_BOHR,
                z * ANGSTROM_TO_BOHR,
                charge
            );
        };

        return line(1.0, 2.0, 3.0, 0.5) + line(4.0, 5.0, 6.0, -0.8);
    }
}   // namespace

class QMWriterTest : public ExternalQMRunnerTest
{
   protected:
    void SetUp() override
    {
        ExternalQMRunnerTest::SetUp();

        // the fixture's atom "H" sits at the origin; add an O and a second H
        addAtom("O", {1.5, 2.25, -3.0});
        addAtom("H", {0.5, 0.5, 0.5});
        _simulationBox.setBoxDimensions({10.0, 12.0, 14.0});
    }

    void addAtom(const std::string &name, const linalg::Vec3D &position)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setName(name);
        atom->setPosition(position);
        _simulationBox.addAtom(atom);
    }

    void addMolecule(
        const molsys::HybridZone zone,
        const bool               active,
        const linalg::Vec3D     &position,
        const double             charge
    )
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setName("X");
        atom->setPosition(position);
        atom->setPartialCharge(charge);

        auto molecule = molsys::Molecule();
        molecule.addAtom(atom);
        molecule.setHybridZone(zone);

        if (!active)
            molecule.deactivateMolecule();

        _simulationBox.addMolecule(molecule);
    }

    void addPointChargeMolecules()
    {
        using enum molsys::HybridZone;

        addMolecule(SMOOTHING, false, {1.0, 2.0, 3.0}, 0.5);
        addMolecule(POINT_CHARGE, false, {4.0, 5.0, 6.0}, -0.8);
        addMolecule(OUTER, false, {7.0, 8.0, 9.0}, 0.1);
        addMolecule(POINT_CHARGE, true, {1.0, 1.0, 1.0}, 0.3);
    }
};

TEST_F(QMWriterTest, dftbCoordsOfAnIsolatedSystemUseTheClusterFlag)
{
    _dftbRunner.writeCoordsFile(_simulationBox);

    EXPECT_EQ(NON_PERIODIC_COORDS, readFile("coords"));
}

TEST_F(QMWriterTest, dftbCoordsOfAPeriodicSystemAppendTheCell)
{
    auto runner = PeriodicDftbRunner();

    runner.writeCoordsFile(_simulationBox);

    auto expected = NON_PERIODIC_COORDS;
    expected.replace(0, 4, "3  S");
    EXPECT_EQ(expected + CELL_LINES, readFile("coords"));
}

TEST_F(QMWriterTest, dftbPointChargesAreOnlyTheInactiveSmoothingAndPointCharge)
{
    addPointChargeMolecules();

    _dftbRunner.writePointChargeFile(_simulationBox);

    EXPECT_EQ(
        POINT_CHARGES,
        readFile(settings::FileSettings::getPointChargeFileName())
    );
}

TEST_F(QMWriterTest, dftbWithoutPointChargesLeavesNoFile)
{
    addMolecule(molsys::HybridZone::OUTER, false, {7.0, 8.0, 9.0}, 0.1);

    _dftbRunner.writePointChargeFile(_simulationBox);

    EXPECT_FALSE(
        std::filesystem::exists(
            settings::FileSettings::getPointChargeFileName()
        )
    );
}

TEST_F(QMWriterTest, dftbEnablesPointChargesOnlyForTheExecutionThatWroteThem)
{
    auto       runner = CommandCaptureRunner<QM::DFTBPlusRunner>();
    const auto path   = _configureQuotedScript(runner);
    const auto suffix = [](const int usePointCharges)
    {
        return std::format(
            " {} {} {}",
            usePointCharges,
            utilities::shellQuote(settings::FileSettings::getDFTBFileName()),
            utilities::shellQuote(
                settings::FileSettings::getPointChargeFileName()
            )
        );
    };

    addPointChargeMolecules();
    runner.writePointChargeFile(_simulationBox);
    runner.execute(_simulationBox);
    EXPECT_TRUE(runner.getCommand().ends_with(suffix(1)))
        << runner.getCommand();

    runner.execute(_simulationBox);
    EXPECT_TRUE(runner.getCommand().ends_with(suffix(0)))
        << runner.getCommand();
    EXPECT_TRUE(
        runner.getCommand().starts_with(utilities::shellQuote(path.string()))
    );
}

TEST_F(QMWriterTest, turbomoleCoordsAreInBohr)
{
    auto runner = QM::TurbomoleRunner();

    runner.writeCoordsFile(_simulationBox);

    const auto line = [](const double       x,
                         const double       y,
                         const double       z,
                         const std::string &name)
    {
        return std::format(
            "   {:16.12f}   {:16.12f}   {:16.12f}   {}\n",
            x * ANGSTROM_TO_BOHR,
            y * ANGSTROM_TO_BOHR,
            z * ANGSTROM_TO_BOHR,
            name
        );
    };

    EXPECT_EQ(
        "$coord\n" + line(0.0, 0.0, 0.0, "H") + line(1.5, 2.25, -3.0, "O") +
            line(0.5, 0.5, 0.5, "H") + "$end\n",
        readFile("coord")
    );
}

TEST_F(QMWriterTest, turbomolePointChargesAreInBohrAndOnlyTheOuterShells)
{
    auto runner = QM::TurbomoleRunner();
    addPointChargeMolecules();

    runner.writePointChargeFile(_simulationBox);

    EXPECT_EQ(
        expectedTurbomolePointCharges(),
        readFile(settings::FileSettings::getPointChargeFileName())
    );
}

TEST_F(QMWriterTest, turbomoleWithoutPointChargesLeavesNoFile)
{
    auto runner = QM::TurbomoleRunner();

    runner.writePointChargeFile(_simulationBox);

    EXPECT_FALSE(
        std::filesystem::exists(
            settings::FileSettings::getPointChargeFileName()
        )
    );
}

TEST_F(QMWriterTest, pyscfCoordsAreAnXyzFile)
{
    auto runner = QM::PySCFRunner();

    runner.writeCoordsFile(_simulationBox);

    EXPECT_EQ(
        "3\n\n"
        "H    \t  0.000000000000\t  0.000000000000\t  0.000000000000\n"
        "O    \t  1.500000000000\t  2.250000000000\t -3.000000000000\n"
        "H    \t  0.500000000000\t  0.500000000000\t  0.500000000000\n",
        readFile("coords.xyz")
    );
}
