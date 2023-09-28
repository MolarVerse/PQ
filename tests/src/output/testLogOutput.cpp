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

#include "testLogOutput.hpp"

#include "systemInfo.hpp"   // for _AUTHOR_, _EMAIL_

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <format>          // for format
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing header to log file
 *
 */
TEST_F(TestLogOutput, writeHeader)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeHeader();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                            _                                    ___   *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*          _                ( )                                 /'___)  *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | |                                    | |                           *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  (_)                                    (_)                           *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, std::format("         Author: {}", sysinfo::_AUTHOR_));
    getline(file, line);
    EXPECT_EQ(line, std::format("         Email:  {}", sysinfo::_EMAIL_));
}

/**
 * @brief tests writing ended normally message to log file
 *
 */
TEST_F(TestLogOutput, writeEndedNormally)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeEndedNormally(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         Elapsed time = 0.1 s");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "INFO:    For citation please refer to the \".ref\" file.");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                      pimd-qmcf ended normally                         *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
}

/**
 * @brief tests writing density warning to log file
 *
 */
TEST_F(TestLogOutput, writeDensityWarning)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeDensityWarning();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "WARNING: Density and box dimensions set. Density will be ignored.");
}

/**
 * @brief tests writing initial momentum to log file
 *
 */
TEST_F(TestLogOutput, writeInitialMomentum)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeInitialMomentum(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    getline(file, line);
    EXPECT_EQ(line, "INFO:    Initial momentum = 0.1 Angstrom * amu / fs");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}