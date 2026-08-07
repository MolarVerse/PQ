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

#include <string>   // for allocator, basic_string

#include "berendsenManostat.hpp"   // for BerendsenManostat
#include "exceptions.hpp"          // for InputFileException, customException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "manostat.hpp"            // for BerendsenManostat, Manostat
#include "manostatSettings.hpp"    // for ManostatSettings
#include "manostatSetup.hpp"       // for ManostatSetup, setupManostat, setup
#include "mdEngine.hpp"            // for MDEngine
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat
#include "testSetup.hpp"                     // for TestSetup
#include "throwWithMessage.hpp"              // for throwWithMessage

using namespace setup;
using namespace settings;
using namespace manostat;

TEST_F(TestSetup, setupManostatNone)
{
    ManostatSetup manostatSetup(*_mdEngine);
    manostatSetup.setup();

    auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::NONE);
    EXPECT_EQ(manostat.getIsotropy(), Isotropy::NONE);
}

TEST_F(TestSetup, setupManostatPressureMissing)
{
    ManostatSetup manostatSetup(*_mdEngine);

    ManostatSettings::setManostatType("berendsen");
    EXPECT_THROW_MSG(
        manostatSetup.setup(),
        customException::InputFileException,
        "Pressure not set for berendsen manostat"
    );
}

TEST_F(TestSetup, setupManostatBerendsen)
{
    ManostatSettings::setManostatType("berendsen");
    ManostatSettings::setIsotropy("isotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    const auto berendsen = dynamic_cast<const BerendsenManostat &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), Isotropy::ISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSemiIsotropicBerendsen)
{
    ManostatSettings::setManostatType("berendsen");
    ManostatSettings::setIsotropy("semi_isotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using SEMI           = SemiIsotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicBerendsen)
{
    ManostatSettings::setManostatType("berendsen");
    ManostatSettings::setIsotropy("anisotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using ANISO          = AnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), Isotropy::ANISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicBerendsen)
{
    ManostatSettings::setManostatType("berendsen");
    ManostatSettings::setIsotropy("full_anisotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::BERENDSEN);

    using FULL_ANISO     = FullAnisotropicBerendsenManostat;
    const auto berendsen = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(berendsen.getIsotropy(), Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(berendsen.getTau(), 0.2 * 1000);
    EXPECT_EQ(berendsen.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSStochasticRescaling)
{
    ManostatSettings::setManostatType(ManostatType::STOCHASTIC_RESCALING);
    ManostatSettings::setIsotropy("isotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using Stochastic      = StochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const Stochastic &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), Isotropy::ISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatSemiIsotropicSStochasticRescaling)
{
    ManostatSettings::setManostatType(ManostatType::STOCHASTIC_RESCALING);
    ManostatSettings::setIsotropy("semi_isotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using SEMI            = SemiIsotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const SEMI &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), Isotropy::SEMI_ISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatAnisotropicSStochasticRescaling)
{
    ManostatSettings::setManostatType(ManostatType::STOCHASTIC_RESCALING);
    ManostatSettings::setIsotropy("anisotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using ANISO           = AnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const ANISO &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), Isotropy::ANISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}

TEST_F(TestSetup, setupManostatFullAnisotropicSStochasticRescaling)
{
    ManostatSettings::setManostatType(ManostatType::STOCHASTIC_RESCALING);
    ManostatSettings::setIsotropy("full_anisotropic");
    ManostatSettings::setPressureSet(true);
    ManostatSettings::setTargetPressure(300.0);
    ManostatSettings::setTauManostat(0.2);
    ManostatSettings::setCompressibility(4.0);

    EXPECT_NO_THROW(setupManostat(*_mdEngine));

    const auto &manostat = _mdEngine->getManostat();
    EXPECT_EQ(manostat.getManostatType(), ManostatType::STOCHASTIC_RESCALING);

    using FULL_ANISO      = FullAnisotropicStochasticRescalingManostat;
    const auto stochastic = dynamic_cast<const FULL_ANISO &>(manostat);
    EXPECT_EQ(stochastic.getIsotropy(), Isotropy::FULL_ANISOTROPIC);
    EXPECT_EQ(stochastic.getTau(), 0.2 * 1000);
    EXPECT_EQ(stochastic.getCompressibility(), 4.0);
}