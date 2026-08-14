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
 * @brief tests parsing an input file name
 */
TEST(TestCommandLineArgs, parse_input_file)
{
    std::vector<std::string> args = {"program", "input.in"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ("input.in", commandLineArgs.getInputFileName());
    EXPECT_EQ(CommandLineAction::RUN, commandLineArgs.getAction());
}

/**
 * @brief tests parsing the help option
 */
TEST(TestCommandLineArgs, parse_help)
{
    for (const auto &option : {"-h", "--help"})
    {
        std::vector<std::string> args = {"program", option};
        auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

        commandLineArgs.parse();
        EXPECT_EQ(CommandLineAction::HELP, commandLineArgs.getAction());
    }
}

/**
 * @brief tests parsing the version option
 */
TEST(TestCommandLineArgs, parse_version)
{
    for (const auto &option : {"-V", "--version"})
    {
        std::vector<std::string> args = {"program", option};
        auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

        commandLineArgs.parse();
        EXPECT_EQ(CommandLineAction::VERSION, commandLineArgs.getAction());
    }
}

/**
 * @brief tests parsing the machine-readable capabilities option
 */
TEST(TestCommandLineArgs, parse_capabilities)
{
    std::vector<std::string> args = {"program", "--capabilities=json"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ(CommandLineAction::CAPABILITIES, commandLineArgs.getAction());
}

/**
 * @brief tests parsing input validation
 */
TEST(TestCommandLineArgs, parse_validation)
{
    std::vector<std::string> args = {"program", "--validate", "input.in"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ(CommandLineAction::VALIDATE, commandLineArgs.getAction());
    EXPECT_EQ(CommandLineFormat::TEXT, commandLineArgs.getFormat());
    EXPECT_EQ(ValidationScope::INSTALLED, commandLineArgs.getValidationScope());
    EXPECT_EQ("input.in", commandLineArgs.getInputFileName());
}

TEST(TestCommandLineArgs, parse_portable_json_validation)
{
    std::vector<std::string> args = {
        "program",
        "--validate",
        "input.in",
        "--scope=portable",
        "--format=json"
    };
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ(CommandLineFormat::JSON, commandLineArgs.getFormat());
    EXPECT_EQ(ValidationScope::PORTABLE, commandLineArgs.getValidationScope());
}

TEST(TestCommandLineArgs, parse_validation_options_in_either_order)
{
    std::vector<std::string> args = {
        "program",
        "--validate",
        "input.in",
        "--format=json",
        "--scope=portable"
    };
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_NO_THROW(commandLineArgs.parse());
    EXPECT_EQ(CommandLineFormat::JSON, commandLineArgs.getFormat());
    EXPECT_EQ(ValidationScope::PORTABLE, commandLineArgs.getValidationScope());
}

/**
 * @brief tests parsing machine-readable input validation
 */
TEST(TestCommandLineArgs, parse_json_validation)
{
    std::vector<std::string> args =
        {"program", "--validate", "input.in", "--format=json"};
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ(CommandLineAction::VALIDATE, commandLineArgs.getAction());
    EXPECT_EQ(CommandLineFormat::JSON, commandLineArgs.getFormat());
    EXPECT_EQ("input.in", commandLineArgs.getInputFileName());
}

TEST(TestCommandLineArgs, parse_explicit_text_validation)
{
    std::vector<std::string> args = {
        "program",
        "--validate",
        "input.in",
        "--format=text",
        "--scope=installed"
    };
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    commandLineArgs.parse();
    EXPECT_EQ(CommandLineFormat::TEXT, commandLineArgs.getFormat());
    EXPECT_EQ(ValidationScope::INSTALLED, commandLineArgs.getValidationScope());
}

TEST(TestCommandLineArgs, reject_duplicate_validation_format)
{
    std::vector<std::string> args =
        {"program", "--validate", "input.in", "--format=text", "--format=json"};
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "Unexpected argument: --format=json. Use PQ --help for usage."
    );
}

/**
 * @brief tests rejecting validation without an input file
 */
TEST(TestCommandLineArgs, parse_validation_without_input)
{
    std::vector<std::string> args = {"program", "--validate"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "No input file specified. Usage: PQ --validate <input_file>"
    );
}

/**
 * @brief tests rejecting a validation format without an input file
 */
TEST(TestCommandLineArgs, parse_validation_format_without_input)
{
    std::vector<std::string> args = {"program", "--validate", "--format=json"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "No input file specified. Usage: PQ --validate <input_file>"
    );
}

/**
 * @brief tests rejecting unsupported validation formats
 */
TEST(TestCommandLineArgs, parse_validation_unknown_format)
{
    std::vector<std::string> args =
        {"program", "--validate", "input.in", "--format=yaml"};
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "Unexpected argument: --format=yaml. Use PQ --help for usage."
    );
}

TEST(TestCommandLineArgs, parse_validation_unknown_scope)
{
    std::vector<std::string> args =
        {"program", "--validate", "input.in", "--scope=project"};
    auto commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "Unexpected argument: --scope=project. Use PQ --help for usage."
    );
}

/**
 * @brief tests rejecting an unknown option
 */
TEST(TestCommandLineArgs, parse_unknown_option)
{
    std::vector<std::string> args = {"program", "--unknown"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "Unknown option: --unknown. Use PQ --help for usage."
    );
}

/**
 * @brief tests throwing exception if no input file name is given
 *
 */
TEST(TestCommandLineArgs, parse_missing_input_file)
{
    std::vector<std::string> args = {"program"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "No input file specified. Usage: PQ <input_file>"
    );
}

/**
 * @brief tests rejecting extra positional arguments
 */
TEST(TestCommandLineArgs, parse_extra_argument)
{
    std::vector<std::string> args = {"program", "input.in", "extra"};
    auto commandLineArgs          = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(
        commandLineArgs.parse(),
        customException::UserInputException,
        "Unexpected argument: extra. Use PQ --help for usage."
    );
}
