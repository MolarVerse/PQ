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

#include "atom.hpp"
#include "berendsenManostat.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostat.hpp"
#include "manostatSettings.hpp"
#include "manostatSetup.hpp"
#include "mdEngine.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"
#include "stochasticRescalingManostat.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

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
