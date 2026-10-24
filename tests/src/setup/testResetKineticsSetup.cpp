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

#include "generalSettings.hpp"
#include "mdEngine.hpp"
#include "resetKineticsSetup.hpp"
#include "testSetup.hpp"
#include "timingsSettings.hpp"

TEST_F(TestSetup, setupResetKineticsIsNoOpWhenNotMDJob)
{
    settings::GeneralSettings::setJobtype(JobType::MM_OPT);
    EXPECT_NO_THROW(setup::setupResetKinetics(*_engine));
}

TEST_F(TestSetup, setupResetKineticsPopulatesResetKineticsOnMDEngine)
{
    settings::GeneralSettings::setJobtype(JobType::MM_MD);
    settings::TimingsSettings::setNumberOfSteps(100);

    EXPECT_NO_THROW(setup::setupResetKinetics(*_mdEngine));
    EXPECT_NO_THROW((void) _mdEngine->getResetKinetics());
}

TEST_F(TestSetup, setupConvertsZeroFrequenciesToNumberOfStepsPlusOne)
{
    settings::GeneralSettings::setJobtype(JobType::MM_MD);
    settings::TimingsSettings::setNumberOfSteps(42);

    setup::ResetKineticsSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());
}

TEST_F(TestSetup, setupAcceptsNonZeroFrequencies)
{
    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    setup::ResetKineticsSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());
}
