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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS

#include <string>   // for allocator, basic_string

#include "exceptions.hpp"                   // for InputFileException
#include "gtest/gtest.h"                    // for Message, TestPartResult
#include "qmmdEngine.hpp"                   // for QMMDEngine
#include "randomNumberGeneratorSetup.hpp"   // for randomNumberGeneratorSetup
#include "settings.hpp"                     // for Settings
#include "throwWithMessage.hpp"             // for ASSERT_THROW_MSG

using setup::RandomNumberGeneratorSetup;
using namespace settings;

TEST(TestRandomNumberGeneratorSetup, setupWithoutRandomSeed)
{
    engine::QMMDEngine engine;
    auto setupRandomNumberGenerator = setup::RandomNumberGeneratorSetup(engine);
    engine.getEngineOutput().getLogOutput().setFilename("default.log");

    Settings::setIsRandomSeedSet(false);
    Settings::setRandomSeed(1);

    setupRandomNumberGenerator.setup();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         Using system-generated random seed");

    ::remove("default.log");
}

TEST(TestRandomNumberGeneratorSetup, setupWithRandomSeed)
{
    engine::QMMDEngine engine;
    auto setupRandomNumberGenerator = setup::RandomNumberGeneratorSetup(engine);
    engine.getEngineOutput().getLogOutput().setFilename("default.log");

    Settings::setIsRandomSeedSet(true);
    Settings::setRandomSeed(73);

    setupRandomNumberGenerator.setup();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         Random seed has been set to: 73");

    ::remove("default.log");
}