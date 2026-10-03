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
#include "ringPolymerSettings.hpp"
#include "ringPolymerSetup.hpp"
#include "ringPolymerqmmdEngine.hpp"
#include "testSetup.hpp"

TEST_F(TestSetup, setupRingPolymerIsNoOpWhenNotActivated)
{
    settings::GeneralSettings::setIsRingPolymerMDActivated(false);
    EXPECT_NO_THROW(setup::setupRingPolymer(*_engine));
}

TEST_F(TestSetup, ringPolymerSetupPhysicalDataResizesBeads)
{
    settings::RingPolymerSettings::setNumberOfBeads(4);
    engine::RingPolymerQMMDEngine rpEngine;

    setup::RingPolymerSetup setup(rpEngine);
    EXPECT_NO_THROW(setup.setupPhysicalData());
}

TEST_F(TestSetup, ringPolymerSetupSimulationBoxAddsBeadsToEngine)
{
    settings::RingPolymerSettings::setNumberOfBeads(3);
    engine::RingPolymerQMMDEngine rpEngine;

    setup::RingPolymerSetup setup(rpEngine);
    EXPECT_NO_THROW(setup.setupSimulationBox());
}
