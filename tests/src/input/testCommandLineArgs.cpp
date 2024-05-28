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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogl...

#include <string>   // for allocator, basic_string, operator+
#include <vector>   // for vector

#include "commandLineArgs.hpp"    // for CommandLineArgs
#include "exceptions.hpp"         // for UserInputException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

/**
 * @brief tests detecting flags and input file name via console input
 *
 */
TEST(TestCommandLineArgs, detectFlags)
{
    std::vector<std::string> args = {"program", "input.in"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    commandLineArgs.detectFlags();
    EXPECT_EQ("input.in", commandLineArgs.getInputFileName());
}

/**
 * @brief tests detecting flags and input file name via console input
 *
 * @TODO: no flags implemented at the moment
 */
TEST(TestCommandLineArgs, detectFlags_flag_given)
{
    std::vector<std::string> args = {"program", "-i", "input.in"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.detectFlags(),
        customException::UserInputException,
        "Invalid flag: " + args[1] + " Flags are not yet implemented."
    );
}

/**
 * @brief tests throwing exception if no input file name is given
 *
 */
TEST(TestCommandLineArgs, detectFlags_missing_input_file)
{
    std::vector<std::string> args = {"program"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.detectFlags(),
        customException::UserInputException,
        "No input file specified. Usage: PQ <input_file>"
    );
}