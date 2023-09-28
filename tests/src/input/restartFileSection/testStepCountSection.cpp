/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "engine.hpp"                   // for Engine
#include "exceptions.hpp"               // for RstFileException, customException
#include "restartFileSection.hpp"       // for RstFileSection, readInput
#include "testRestartFileSection.hpp"   // for TestStepCountSection
#include "timings.hpp"                  // for Timings

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPart...
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F, InitG...
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace input;

TEST_F(TestStepCountSection, testKeyword) { EXPECT_EQ(_section->keyword(), "step"); }

TEST_F(TestStepCountSection, testIsHeader) { EXPECT_TRUE(_section->isHeader()); }

TEST_F(TestStepCountSection, testNumberOfArguments)
{
    for (size_t i = 0; i < 10; ++i)
        if (i != 2)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);
        }
}

TEST_F(TestStepCountSection, testNegativeStepCount)
{
    auto line = std::vector<std::string>(2);
    line[1]   = "-1";
    ASSERT_THROW(_section->process(line, *_engine), customException::RstFileException);
}

TEST_F(TestStepCountSection, testProcess)
{
    auto line = std::vector<std::string>(2);
    line[1]   = "1000";
    _section->process(line, *_engine);
    EXPECT_EQ(_engine->getTimings().getStepCount(), 1000);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}