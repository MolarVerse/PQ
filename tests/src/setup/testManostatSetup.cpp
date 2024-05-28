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
#include "engine.hpp"              // for Engine
#include "exceptions.hpp"          // for InputFileException, customException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "manostat.hpp"            // for BerendsenManostat, Manostat
#include "manostatSettings.hpp"    // for ManostatSettings
#include "manostatSetup.hpp"       // for ManostatSetup, setupManostat, setup
#include "testSetup.hpp"           // for TestSetup

using namespace setup;

/**
 * @TODO: refactor this test to use the new setupManostat function
 *
 * @TODO: include compressibility in the test
 *
 */
TEST_F(TestSetup, setup)
{
    ManostatSetup manostatSetup(*_engine);
    manostatSetup.setup();

    settings::ManostatSettings::setManostatType("berendsen");
    EXPECT_THROW(manostatSetup.setup(), customException::InputFileException);

    settings::ManostatSettings::setPressureSet(true);
    settings::ManostatSettings::setTargetPressure(300.0);
    EXPECT_NO_THROW(manostatSetup.setup());

    const auto berendsenManostat =
        dynamic_cast<manostat::BerendsenManostat &>(_engine->getManostat());
    EXPECT_EQ(berendsenManostat.getTau(), 1.0 * 1000);

    settings::ManostatSettings::setTauManostat(0.2);
    EXPECT_NO_THROW(manostatSetup.setup());

    const auto berendsenManostat2 =
        dynamic_cast<manostat::BerendsenManostat &>(_engine->getManostat());
    EXPECT_EQ(berendsenManostat2.getTau(), 0.2 * 1000);

    EXPECT_NO_THROW(setupManostat(*_engine));
}