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

#include <gtest/gtest.h>   // for EXPECT_EQ, EXPECT_NO_THROW, InitGoog...

#include "berendsenManostat.hpp"   // for manostat::BerendsenManostat
#include "exceptions.hpp"          // for UserInputException
#include "manostat.hpp"            // for manostat::BerendsenManostat, Manostat
#include "manostatSettings.hpp"    // for settings::ManostatSettings
#include "manostatSetup.hpp"   // for setup::ManostatSetup, setupManostat, setup
#include "mdEngine.hpp"        // for MDEngine
#include "settings.hpp"        // for JobType, settings::Settings
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat
#include "testSetup.hpp"                     // for TestSetup
#include "throwWithMessage.hpp"              // for EXPECT_THROW_MSG

TEST_F(TestSetup, setupManostatSkipsNonMDJobs)
{
    const auto jobType = settings::Settings::getJobtype();
    settings::Settings::setJobtype(settings::JobType::MM_OPT);

    EXPECT_NO_THROW(setup::setupManostat(*_engine));

    settings::Settings::setJobtype(jobType);
}

TEST_F(TestSetup, setupManostatNone)
{
    setup::ManostatSetup manostatSetup(*_mdEngine);
    manostatSetup.setup();

    auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::NONE);
    EXPECT_EQ(manostat.getIsotropy(), settings::Isotropy::NONE);
}

TEST_F(TestSetup, setupManostatBerendsen)
{
    settings::ManostatSettings::setManostatType("berendsen");
    settings::ManostatSettings::setIsotropy("isotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);

    const auto berendsen =
        dynamic_cast<const manostat::BerendsenManostat &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), settings::Isotropy::ISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatNoneIsotropyDefaultsToIsotropic)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    settings::ManostatSettings::setIsotropy(settings::Isotropy::NONE);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    const auto &manostat = _mdEngine->getManostat();
    const auto  berendsen =
        dynamic_cast<const manostat::BerendsenManostat &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), settings::Isotropy::ISOTROPIC);
}

TEST_F(TestSetup, setupManostatSemiIsotropicBerendsen)
{
    settings::ManostatSettings::setManostatType("berendsen");
    settings::ManostatSettings::setIsotropy("semi_isotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);

    using SEMI           = manostat::SemiIsotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), settings::Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicBerendsen)
{
    settings::ManostatSettings::setManostatType("berendsen");
    settings::ManostatSettings::setIsotropy("anisotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);

    using ANISO          = manostat::AnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), settings::Isotropy::ANISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicBerendsen)
{
    settings::ManostatSettings::setManostatType("berendsen");
    settings::ManostatSettings::setIsotropy("full_anisotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);

    using FULL_ANISO     = manostat::FullAnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), settings::Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy("isotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(
        manostat.getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    using Stochastic      = manostat::StochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const Stochastic &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), settings::Isotropy::ISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSemiIsotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy("semi_isotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(
        manostat.getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    using SEMI            = manostat::SemiIsotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), settings::Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy("anisotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(
        manostat.getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    using ANISO           = manostat::AnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), settings::Isotropy::ANISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicSStochasticRescaling)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    settings::ManostatSettings::setIsotropy("full_anisotropic");
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(
        manostat.getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    using FULL_ANISO = manostat::FullAnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), settings::Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, validateIsotropyFixedAxisCombinationSemiIsotropic)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    settings::ManostatSettings::setIsotropy(settings::Isotropy::SEMI_ISOTROPIC);
    // Default semi-isotropic axes in parser are xy (anisotropic axis = z)
    settings::ManostatSettings::set2DIsotropicAxes({0U, 1U});
    settings::ManostatSettings::set2DAnisotropicAxis(2U);

    // Fixing Z (anisotropic axis) is allowed
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::Z);
    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_NO_THROW(manostatSetup.setup());

    // Fixing X or Y or XY should throw
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::X);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::XY);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: semi-isotropic pressure coupling only "
        "allows fixing the anisotropic axis or none."
    );

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::NONE);
}

TEST_F(TestSetup, validateFixedAxisThrowsWhenAllAxesFixedWithManostat)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    settings::ManostatSettings::setIsotropy(settings::Isotropy::ISOTROPIC);
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::ALL);

    setup::ManostatSetup manostatSetup(*_mdEngine);
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        exc::UserInputException,
        "Invalid combination: all axes cannot be fixed while a "
        "manostat is selected."
    );

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::NONE);
}

TEST_F(TestSetup, setupManostatWithFixedAxis)
{
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    settings::ManostatSettings::setIsotropy(settings::Isotropy::ISOTROPIC);
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::XY);
    settings::ManostatSettings::setTargetPressure(300.0);
    settings::ManostatSettings::setTauManostat(0.2);
    settings::ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setup::setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), settings::ManostatType::BERENDSEN);

    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::NONE);
}
