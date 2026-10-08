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

#include "berendsenManostat.hpp"
#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostat.hpp"
#include "manostatSettings.hpp"
#include "manostatSetup.hpp"
#include "mdEngine.hpp"
#include "stochasticRescalingManostat.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"

TEST_F(TestSetup, setupManostatSkipsNonMDJobs)
{
    const auto jobType = settings::GeneralSettings::getJobtype();
    settings::GeneralSettings::setJobtype(JobType::MM_OPT);

    EXPECT_NO_THROW(setup::setupManostat(*_engine));

    settings::GeneralSettings::setJobtype(jobType);
}

TEST_F(TestSetup, setupManostatNone)
{
    setup::ManostatSetup manostatSetup(*_mdEngine);
    manostatSetup.setup();

    auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::NONE);
}

TEST_F(TestSetup, setupManostatBerendsen)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    const auto berendsen =
        dynamic_cast<const manostat::BerendsenManostat &>(manostat);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatNoneIsotropyDefaultsToIsotropic)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    const auto &manostat = _mdEngine->getManostat();
    const auto  berendsen =
        dynamic_cast<const manostat::BerendsenManostat &>(manostat);
}

TEST_F(TestSetup, setupManostatSemiIsotropicBerendsen)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using SEMI           = manostat::SemiIsotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicBerendsen)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::ANISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using ANISO          = manostat::AnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicBerendsen)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::FULL_ANISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using FULL_ANISO     = manostat::FullAnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using Stochastic      = manostat::StochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const Stochastic &>(manostat);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSemiIsotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using SEMI            = manostat::SemiIsotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy(Isotropy::ANISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using ANISO           = manostat::AnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy(Isotropy::FULL_ANISOTROPIC);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using FULL_ANISO = manostat::FullAnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, validateIsotropyFixedAxisCombinationSemiIsotropic)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::SEMI_ISOTROPIC_XY);

    // Fixing Z (anisotropic axis) is allowed
    settings::ManostatSettings::setFixedAxis(FixedAxis::Z);
    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    // Fixing X or Y or XY should throw
    settings::ManostatSettings::setFixedAxis(FixedAxis::X);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    settings::ManostatSettings::setFixedAxis(FixedAxis::XY);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    settings::ManostatSettings::setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, validateFixedAxisThrowsWhenAllAxesFixedWithManostat)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setFixedAxis(FixedAxis::ALL);

    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    settings::ManostatSettings::setManostatType(
        ManostatType::STOCHASTIC_RESCALING
    );
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    settings::ManostatSettings::setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, setupManostatWithFixedAxis)
{
    settings::ManostatSettings::setManostatType(ManostatType::BERENDSEN);
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setFixedAxis(FixedAxis::XY);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    settings::ManostatSettings::setFixedAxis(FixedAxis::NONE);
}

TEST_F(TestSetup, setupMolecularManostatsRejectAtomicVirial)
{
    settings::GeneralSettings::setVirialType(VirialType::ATOMIC);
    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    settings::ManostatSettings::setFixedAxis(FixedAxis::NONE);
    for (const auto type :
         {ManostatType::BERENDSEN, ManostatType::STOCHASTIC_RESCALING})
    {
        settings::ManostatSettings::setManostatType(type);
        setup::ManostatSetup setup(*_mdEngine);
        EXPECT_THROW(setup.setup(), exc::UserInputException);
    }
    settings::ManostatSettings::setManostatType(ManostatType::NONE);
    setup::ManostatSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());
    settings::GeneralSettings::setVirialType(VirialType::MOLECULAR);
}
