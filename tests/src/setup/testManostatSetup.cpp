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

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atom.hpp"
#include "berendsenManostat.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "logOutput.hpp"
#include "manostat.hpp"
#include "manostatSettings.hpp"
#include "manostatSetup.hpp"
#include "mdEngine.hpp"
#include "molecule.hpp"
#include "outputMessages.hpp"
#include "simulationBox.hpp"
#include "stochasticRescalingManostat.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

namespace
{
    // gives the tests access to the static write functions of ManostatSetup
    class TestableManostatSetup : public setup::ManostatSetup
    {
       public:
        static void writeManostatSelection(
            out::LogOutput         &logOutput,
            const ManostatSettings &settings
        )
        {
            _writeManostatSelection(logOutput, settings);
        }

        static void writeBerendsenSetup(
            out::LogOutput         &logOutput,
            const ManostatSettings &settings
        )
        {
            _writeBerendsenSetup(logOutput, settings);
        }

        static void writeIsotropy(
            out::LogOutput         &logOutput,
            const ManostatSettings &settings
        )
        {
            _writeIsotropy(logOutput, settings);
        }
    };

    /**
     * @brief what a write function puts into a real log file, line by line
     */
    template <typename Writer>
    std::vector<std::string> writtenLines(Writer &&writer)
    {
        const std::string fileName     = "manostat-setup-write.log";
        const auto        removeResult = std::remove(fileName.c_str());
        if (removeResult != 0 && errno != ENOENT)
        {
            throw std::runtime_error(
                "Failed to remove existing log file: " + fileName
            );
        }

        out::LogOutput logOutput(fileName);
        logOutput.setFilename(fileName);
        std::forward<Writer>(writer)(logOutput);
        logOutput.close();

        std::ifstream            file(fileName);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) lines.push_back(line);
        file.close();

        EXPECT_EQ(std::remove(fileName.c_str()), 0);
        return lines;
    }

    // a line as the log writes it: setup info is indented by out::OUTPUT
    std::string info(const std::string &text) { return out::OUTPUT + text; }
}   // namespace

TEST_F(TestSetup, setupManostatSkipsNonMDJobs)
{
    const auto jobType = settings::GeneralSettings::getJobtype();
    settings::GeneralSettings::setJobtype(JobType::MM_OPT);

    EXPECT_NO_THROW(setup::setupManostat(*_engine));

    settings::GeneralSettings::setJobtype(jobType);
}

TEST_F(TestSetup, setupManostatNone)
{
    _settings->finalize();

    setup::ManostatSetup manostatSetup(*_mdEngine);
    manostatSetup.setup();

    EXPECT_EQ(
        _settings->manostatSettings.getManostatType(),
        ManostatType::NONE
    );
}

TEST_F(TestSetup, setupManostatBerendsen)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    // The manostat's concrete type is picked by the factory when the engine
    // is constructed, so settings must be configured before construction.
    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    EXPECT_NO_THROW((void
    ) dynamic_cast<const manostat::BerendsenManostat &>(manostat));
}

TEST_F(TestSetup, setupManostatNoneIsotropyDefaultsToIsotropic)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine   mdEngine(*_settings);
    setup::ManostatSetup manostatSetup(mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    const auto &manostat = mdEngine.getManostat();
    EXPECT_NO_THROW((void
    ) dynamic_cast<const manostat::BerendsenManostat &>(manostat));
}

TEST_F(TestSetup, setupManostatSemiIsotropicBerendsen)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using SEMI           = manostat::SemiIsotropicBerendsenManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const SEMI &>(manostat));
}

TEST_F(TestSetup, setupManostatAnisotropicBerendsen)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::ANISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using ANISO          = manostat::AnisotropicBerendsenManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const ANISO &>(manostat));
}

TEST_F(TestSetup, setupManostatFullAnisotropicBerendsen)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::FULL_ANISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using FULL_ANISO     = manostat::FullAnisotropicBerendsenManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const FULL_ANISO &>(manostat));
}

TEST_F(TestSetup, setupManostatSStochasticRescaling)
{
    _settings->manostatSettings.setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using Stochastic     = manostat::StochasticRescalingManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const Stochastic &>(manostat));
}

TEST_F(TestSetup, setupManostatSemiIsotropicSStochasticRescaling)
{
    _settings->manostatSettings.setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    _settings->manostatSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using SEMI           = manostat::SemiIsotropicStochasticRescalingManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const SEMI &>(manostat));
}

TEST_F(TestSetup, setupManostatAnisotropicSStochasticRescaling)
{
    _settings->manostatSettings.setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    _settings->manostatSettings.setIsotropy(Isotropy::ANISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using ANISO          = manostat::AnisotropicStochasticRescalingManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const ANISO &>(manostat));
}

TEST_F(TestSetup, setupManostatFullAnisotropicSStochasticRescaling)
{
    _settings->manostatSettings.setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    _settings->manostatSettings.setIsotropy(Isotropy::FULL_ANISOTROPIC);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    using FULL_ANISO     = manostat::FullAnisotropicStochasticRescalingManostat;
    EXPECT_NO_THROW((void) dynamic_cast<const FULL_ANISO &>(manostat));
}

TEST_F(TestSetup, validateIsotropyFixedAxisCombinationSemiIsotropic)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);

    // Fixing Z (anisotropic axis) is allowed
    _settings->manostatSettings.setFixedAxis(FixedAxis::Z);
    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    // Fixing X or Y or XY should throw
    _settings->manostatSettings.setFixedAxis(FixedAxis::X);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    _settings->manostatSettings.setFixedAxis(FixedAxis::XY);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, validateFixedAxisThrowsWhenAllAxesFixedWithManostat)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setFixedAxis(FixedAxis::ALL);

    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    _settings->manostatSettings.setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, setupManostatWithFixedAxis)
{
    _settings->manostatSettings.setManostatType(ManostatType::BERENDSEN);
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setFixedAxis(FixedAxis::XY);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->finalize();

    engine::MMMDEngine mdEngine(*_settings);
    EXPECT_NO_THROW(setup::setupManostat(mdEngine));

    const auto &manostat = mdEngine.getManostat();
    EXPECT_NO_THROW((void
    ) dynamic_cast<const manostat::BerendsenManostat &>(manostat));

    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, setupMolecularManostatsWarnForAtomicVirial)
{
    molsys::Molecule molecule;
    for (size_t i = 0; i < 2; ++i)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setMass(1.0);
        molecule.addAtom(atom);
    }
    _mdEngine->getSimulationBox().addMolecule(molecule);
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->manostatSettings.finalize();
    const auto types = std::array{
        ManostatType::BERENDSEN,
        ManostatType::STOCHASTIC_RESCALING,
        ManostatType::NONE
    };
    for (const auto type : types)
    {
        _settings->manostatSettings.setManostatType(type);
        auto &log = _mdEngine->getLogOutput();
        log.setFilename("atomic-virial-warning.log");
        setup::ManostatSetup setup(*_mdEngine);
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(setup.setup());
        const auto output = testing::internal::GetCapturedStdout();
        log.close();

        std::ifstream     file("atomic-virial-warning.log");
        std::stringstream logContents;
        logContents << file.rdbuf();
        file.close();
        for (const auto &text : {output, logContents.str()})
        {
            const auto warning = text.find("WARNING:");
            if (type == ManostatType::NONE)
            {
                EXPECT_EQ(warning, std::string::npos);
            }
            else
            {
                EXPECT_NE(warning, std::string::npos);
                if (warning != std::string::npos)
                {
                    EXPECT_EQ(
                        text.find("WARNING:", warning + 1),
                        std::string::npos
                    );
                }
                EXPECT_NE(text.find("virial = molecular"), std::string::npos);
                EXPECT_NE(text.find("Continuing"), std::string::npos);
            }
        }
        EXPECT_EQ(std::remove("atomic-virial-warning.log"), 0);
    }
    settings::GeneralSettings::setVirialType(VirialType::MOLECULAR);
}

/* ---------- what the setup writes into the log ---------- */

TEST(TestManostatSetupOutput, selectionNamesTheManostat)
{
    const auto expectedText = std::vector<std::pair<ManostatType, std::string>>{
        {ManostatType::BERENDSEN, "Berendsen manostat selected"},
        {ManostatType::STOCHASTIC_RESCALING,
         "Stochastic rescaling manostat selected"},
        {ManostatType::NONE, "No manostat selected"}
    };

    for (const auto &[type, text] : expectedText)
    {
        SCOPED_TRACE(text);
        ManostatSettings settings;
        settings.setManostatType(type);

        const auto lines = writtenLines(
            [&settings](auto &log)
            { TestableManostatSetup::writeManostatSelection(log, settings); }
        );

        EXPECT_EQ(lines, (std::vector<std::string>{info(text), ""}));
    }
}

TEST(TestManostatSetupOutput, berendsenSetupWritesPressureTauAndCompressibility)
{
    ManostatSettings settings;
    settings.setTargetPressure(300.0);
    settings.setTau(0.2 * ps);
    settings.setCompressibility(4.0);

    const auto lines = writtenLines(
        [&settings](auto &log)
        { TestableManostatSetup::writeBerendsenSetup(log, settings); }
    );

    EXPECT_EQ(
        lines,
        (std::vector<std::string>{
            info("Target pressure: 300"),
            info("Relaxation time: 0.2 ps"),
            info("Compressibility: 4 bar⁻¹"),
            ""
        })
    );
}

TEST(TestManostatSetupOutput, relaxationTimeIsWrittenInPicosecondsForAnyUnit)
{
    ManostatSettings settings;
    settings.setTargetPressure(1.0);   // has no default
    settings.setTau(500.0 * fs);

    const auto lines = writtenLines(
        [&settings](auto &log)
        { TestableManostatSetup::writeBerendsenSetup(log, settings); }
    );

    ASSERT_EQ(lines.size(), 4U);
    EXPECT_EQ(lines[1], info("Relaxation time: 0.5 ps"));
}

TEST(TestManostatSetupOutput, isotropyWritesOneLineForTheSimpleModes)
{
    const auto expectedText = std::vector<std::pair<Isotropy, std::string>>{
        {Isotropy::ISOTROPIC, "Isotropy: isotropic"},
        {Isotropy::ANISOTROPIC, "Isotropy: anisotropic"},
        {Isotropy::FULL_ANISOTROPIC, "Isotropy: full anisotropic"}
    };

    for (const auto &[isotropy, text] : expectedText)
    {
        SCOPED_TRACE(text);
        ManostatSettings settings;
        settings.setIsotropy(isotropy);

        const auto lines = writtenLines(
            [&settings](auto &log)
            { TestableManostatSetup::writeIsotropy(log, settings); }
        );

        EXPECT_EQ(lines, (std::vector<std::string>{info(text), ""}));
    }
}

TEST(TestManostatSetupOutput, semiIsotropicNamesTheAnisotropicAndIsotropicAxes)
{
    struct Case
    {
        Isotropy    isotropy;
        std::string anisotropicAxis;
        std::string isotropicAxes;
    };

    const auto cases = std::vector<Case>{
        {.isotropy        = Isotropy::SEMI_ISOTROPIC_XY,
         .anisotropicAxis = "z",
         .isotropicAxes   = "x, y"},
        {.isotropy        = Isotropy::SEMI_ISOTROPIC_XZ,
         .anisotropicAxis = "y",
         .isotropicAxes   = "x, z"},
        {.isotropy        = Isotropy::SEMI_ISOTROPIC_YZ,
         .anisotropicAxis = "x",
         .isotropicAxes   = "y, z"}
    };

    for (const auto &[isotropy, anisotropicAxis, isotropicAxes] : cases)
    {
        SCOPED_TRACE(isotropicAxes);
        ManostatSettings settings;
        settings.setIsotropy(isotropy);

        const auto lines = writtenLines(
            [&settings](auto &log)
            { TestableManostatSetup::writeIsotropy(log, settings); }
        );

        EXPECT_EQ(
            lines,
            (std::vector<std::string>{
                info("Isotropy:         semi-isotropic"),
                info("Anisotropic axis: " + anisotropicAxis),
                info("Isotropic axes:   " + isotropicAxes),
                ""
            })
        );
    }
}

TEST_F(TestSetup, setupWritesSelectionParametersAndIsotropyInOrder)
{
    _settings->manostatSettings.setIsotropy(Isotropy::ISOTROPIC);
    _settings->manostatSettings.setFixedAxis(FixedAxis::NONE);
    _settings->manostatSettings.setTargetPressure(300.0);
    _settings->manostatSettings.setTau(0.2 * ps);
    _settings->manostatSettings.setCompressibility(4.0);
    _settings->manostatSettings.finalize();

    const auto expectedByType =
        std::vector<std::pair<ManostatType, std::vector<std::string>>>{
            {ManostatType::BERENDSEN,
             {info("Berendsen manostat selected"),
              "",
              info("Target pressure: 300"),
              info("Relaxation time: 0.2 ps"),
              info("Compressibility: 4 bar⁻¹"),
              "",
              info("Isotropy: isotropic"),
              ""}},
            {ManostatType::STOCHASTIC_RESCALING,
             {info("Stochastic rescaling manostat selected"),
              "",
              info("Target pressure: 300"),
              info("Relaxation time: 0.2 ps"),
              info("Compressibility: 4 bar⁻¹"),
              "",
              info("Isotropy: isotropic"),
              ""}},
            {ManostatType::NONE, {info("No manostat selected"), ""}}
        };

    for (const auto &[type, expected] : expectedByType)
    {
        SCOPED_TRACE(static_cast<int>(type));
        _settings->manostatSettings.setManostatType(type);

        const std::string fileName = "manostat-setup-order.log";
        // Remove any existing log file before starting the test to ensure a
        // clean slate. If not present it will simply return a non-zero value,
        // which we ignore here.
        const auto _   = std::remove(fileName.c_str());
        auto      &log = _mdEngine->getLogOutput();
        log.setFilename(fileName);

        setup::ManostatSetup setup(*_mdEngine);
        testing::internal::CaptureStdout();
        EXPECT_NO_THROW(setup.setup());
        testing::internal::GetCapturedStdout();
        log.close();

        std::ifstream            file(fileName);
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) lines.push_back(line);
        file.close();

        EXPECT_EQ(lines, expected);
        EXPECT_EQ(std::remove(fileName.c_str()), 0);
    }
}
