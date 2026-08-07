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

#include "testStdoutOutput.hpp"

#include <format>   // for format
#include <iosfwd>   // for stringstream
#include <string>   // for allocator, string

#include "gtest/gtest.h"        // for Message, TestPartResult
#include "outputMessages.hpp"   // for _OUTPUT_
#include "systemInfo.hpp"       // for _AUTHOR_, _EMAIL_

/**
 * @brief tests writing header to stdout
 *
 */
TEST_F(TestStdoutOutput, writeHeader)
{
    testing::internal::CaptureStdout();
    _stdoutOutput->writeHeader();
    std::string output = testing::internal::GetCapturedStdout();

    std::stringstream sstream(output);
    std::string       line;
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(************************************************************************)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88888888ba     ,ad8888ba,                       *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88      "8b   d8"'    `"8b                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88      ,8P  d8'        `8b                     *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88aaaaaa8P'  88          88                     *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88""""""'    88          88                     *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88           Y8,    "88,,8P                     *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88            Y8a.    Y88P                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                      88             `"Y8888Y"Y8a                     *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                      *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(************************************************************************)"
    );
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(
        line,
        std::format("         Author:        {}", sysinfo::_AUTHOR_)
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        std::format("         Email:         {}", sysinfo::_EMAIL_)
    );
}

/**
 * @brief tests writing ended normally message to stdout
 *
 */
TEST_F(TestStdoutOutput, writeEndedNormally)
{
    testing::internal::CaptureStdout();
    _stdoutOutput->writeEndedNormally(0.1);
    std::string output = testing::internal::GetCapturedStdout();

    std::stringstream sstream(output);
    std::string       line;
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(line, "         Elapsed time = 0.10000 s");
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(line, "INFO:    For citation please refer to the \".ref\" file.");
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*************************************************************************)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                       *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                          PQ ended normally                            *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*                                                                       *)"
    );
    getline(sstream, line);
    EXPECT_EQ(
        line,
        R"(*************************************************************************)"
    );
}

/**
 * @brief Test writeDensityWarning
 *
 */
TEST_F(TestStdoutOutput, writeDensityWarning)
{
    testing::internal::CaptureStdout();
    _stdoutOutput->writeDensityWarning();
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(
        output,
        std::format(
            "{}\x1B[33mUserInputWarning\x1B[39m\n{}Density and box dimensions "
            "set. Density will be ignored.\n\n",
            output::_OUTPUT_,
            output::_OUTPUT_
        )
    );
}