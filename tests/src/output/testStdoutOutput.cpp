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

#include "outputMessages.hpp"   // for _OUTPUT_
#include "systemInfo.hpp"       // for _AUTHOR_, _EMAIL_

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <format>          // for format
#include <iosfwd>          // for stringstream
#include <string>          // for allocator, string

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
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                            _                                    ___   *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*          _                ( )                                 /'___)  *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*  | |                                    | |                           *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*  (_)                                    (_)                           *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(sstream, line);
    EXPECT_EQ(line, "");
    getline(sstream, line);
    EXPECT_EQ(line, std::format("         Author:        {}", sysinfo::_AUTHOR_));
    getline(sstream, line);
    EXPECT_EQ(line, std::format("         Email:         {}", sysinfo::_EMAIL_));
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
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                      PQ ended normally                         *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(sstream, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
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

    EXPECT_EQ(output,
              std::format("{}\x1B[33mUserInputWarning\x1B[39m\n{}Density and box dimensions set. Density will be ignored.\n\n",
                          output::_OUTPUT_,
                          output::_OUTPUT_));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}