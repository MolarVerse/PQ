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

#include <cstddef>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "restartFileReader/restartFileSection.hpp"
#include "testRestartFileSection.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

TEST_F(TestStepCountSection, testKeyword)
{
    EXPECT_EQ(_section->keyword(), "step");
}

TEST_F(TestStepCountSection, testIsHeader)
{
    EXPECT_TRUE(_section->isHeader());
}

TEST_F(TestStepCountSection, testNumberOfArguments)
{
    size_t lineNumber = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        if (i != 2 && i != 4)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW_MSG(
                _section->process(line, *_engine, lineNumber),
                exc::RstFileException,
                "Error in line 0: Step count section must have 2 or 4 elements"
            );
        }
    }
}

TEST_F(TestStepCountSection, testNegativeStepCount)
{
    auto line         = std::vector<std::string>(2);
    line[1]           = "-1";
    size_t lineNumber = 0;
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0: Step count must be positive"
    );
}

TEST_F(TestStepCountSection, testProcess)
{
    auto line         = std::vector<std::string>(2);
    line[1]           = "1000";
    size_t lineNumber = 0;
    _section->process(line, *_engine, lineNumber);
    EXPECT_EQ(_settings.timings.getStartingStepCount(), 1000);
}

TEST_F(TestStepCountSection, testProcessWithSimulationTime)
{
    auto line =
        std::vector<std::string>{"step", "1000", "Simulation_Time", "2.5"};
    size_t lineNumber = 0;
    _section->process(line, *_engine, lineNumber);

    EXPECT_EQ(_settings.timings.getStartingStepCount(), 1000);
    EXPECT_TRUE(_settings.timings.isStartingTimeSet());
    EXPECT_DOUBLE_EQ(
        _settings.timings.getStartingTime().numerical_value_in(ps),
        2.5
    );
}

TEST_F(TestStepCountSection, testWrongSimulationTimeKeyword)
{
    auto   line       = std::vector<std::string>{"step", "1000", "foo", "2.5"};
    size_t lineNumber = 0;
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0: Expected keyword \"simulation_time\" after the step "
        "count but got \"foo\""
    );
}

TEST_F(TestStepCountSection, testInvalidSimulationTime)
{
    auto line =
        std::vector<std::string>{"step", "1000", "simulation_time", "2.5x"};
    size_t lineNumber = 0;
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0: Invalid simulation time \"2.5x\""
    );
}

TEST_F(TestStepCountSection, testNegativeSimulationTime)
{
    auto line =
        std::vector<std::string>{"step", "1000", "simulation_time", "-1"};
    size_t lineNumber = 0;
    ASSERT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0: Simulation time must not be negative"
    );
}
