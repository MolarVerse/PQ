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

#include "commandLineArgs.hpp"

#include "exceptions.hpp"   // for UserInputException

using namespace customException;

/**
 * @brief Construct a new CommandLineArgs::CommandLineArgs object
 *
 * @param argc
 * @param argv
 */
CommandLineArgs::CommandLineArgs(
    const int                       argc,
    const std::vector<std::string> &argv
)
    : _argc(argc), _argv(argv)
{
}

/**
 * @brief Parses the command line arguments.
 *
 * @throw UserInputException if the command line is invalid
 */
void CommandLineArgs::parse()
{
    if (_argc < 2)
        throw UserInputException(
            "No input file specified. Usage: PQ <input_file>"
        );

    const auto &argument = _argv[1];

    if ("--help" == argument || "-h" == argument)
        _action = CommandLineAction::HELP;
    else if ("--version" == argument || "-V" == argument)
        _action = CommandLineAction::VERSION;
    else if ("--capabilities=json" == argument)
        _action = CommandLineAction::CAPABILITIES;
    else if (argument.starts_with('-'))
        throw UserInputException(
            "Unknown option: " + argument + ". Use PQ --help for usage."
        );
    else
        _inputFileName = argument;

    if (_argc > 2)
        throw UserInputException(
            "Unexpected argument: " + _argv[2] + ". Use PQ --help for usage."
        );
}

/**
 * @brief get the input file name
 *
 * @return std::string
 */
std::string CommandLineArgs::getInputFileName() const { return _inputFileName; }

/**
 * @brief get the requested command line action
 *
 * @return CommandLineAction
 */
CommandLineAction CommandLineArgs::getAction() const { return _action; }
