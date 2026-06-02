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

#include "mdEngine.hpp"
#include "resetKineticsSettings.hpp"
#include "resetKineticsSetup.hpp"
#include "settings.hpp"
#include "testSetup.hpp"
#include "timingsSettings.hpp"

using namespace setup::resetKinetics;
using namespace settings;

namespace
{
    void resetSettings()
    {
        ResetKineticsSettings::setNScale(0);
        ResetKineticsSettings::setFScale(0);
        ResetKineticsSettings::setNReset(0);
        ResetKineticsSettings::setFReset(0);
        ResetKineticsSettings::setNResetAngular(0);
        ResetKineticsSettings::setFResetAngular(0);
        ResetKineticsSettings::setFResetForces(0);
    }
}   // namespace

TEST_F(TestSetup, setupResetKineticsIsNoOpWhenNotMDJob)
{
    resetSettings();
    Settings::setJobtype(JobType::MM_OPT);
    EXPECT_NO_THROW(setupResetKinetics(*_engine));
}

TEST_F(TestSetup, setupResetKineticsPopulatesResetKineticsOnMDEngine)
{
    resetSettings();
    Settings::setJobtype(JobType::MM_MD);
    TimingsSettings::setNumberOfSteps(100);

    EXPECT_NO_THROW(setupResetKinetics(*_mdEngine));
    EXPECT_NO_THROW((void)_mdEngine->getResetKinetics());
}

TEST_F(TestSetup, setupConvertsZeroFrequenciesToNumberOfStepsPlusOne)
{
    resetSettings();
    Settings::setJobtype(JobType::MM_MD);
    TimingsSettings::setNumberOfSteps(42);

    ResetKineticsSetup s(*_mdEngine);
    EXPECT_NO_THROW(s.setup());
}

TEST_F(TestSetup, setupAcceptsNonZeroFrequencies)
{
    resetSettings();
    Settings::setJobtype(JobType::MM_MD);
    TimingsSettings::setNumberOfSteps(50);
    ResetKineticsSettings::setFScale(10);
    ResetKineticsSettings::setFReset(5);
    ResetKineticsSettings::setFResetAngular(3);
    ResetKineticsSettings::setFResetForces(2);

    ResetKineticsSetup s(*_mdEngine);
    EXPECT_NO_THROW(s.setup());
}
