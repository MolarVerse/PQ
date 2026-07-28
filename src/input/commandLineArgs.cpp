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

    if ("--validate" == argument)
    {
        _action = CommandLineAction::VALIDATE;

        if (_argc < 3 || _argv[2].starts_with('-'))
            throw UserInputException(
                "No input file specified. Usage: PQ --validate <input_file>"
            );

        _inputFileName = _argv[2];

        auto formatSet = false;
        auto scopeSet  = false;
        for (auto index = 3; index < _argc; ++index)
        {
            const auto &option = _argv[size_t(index)];

            if ("--format=json" == option && !formatSet)
            {
                _format   = CommandLineFormat::JSON;
                formatSet = true;
            }
            else if ("--format=text" == option && !formatSet)
            {
                _format   = CommandLineFormat::TEXT;
                formatSet = true;
            }
            else if ("--scope=installed" == option && !scopeSet)
            {
                _validationScope = ValidationScope::INSTALLED;
                scopeSet         = true;
            }
            else if ("--scope=portable" == option && !scopeSet)
            {
                _validationScope = ValidationScope::PORTABLE;
                scopeSet         = true;
            }
            else
                throw UserInputException(
                    "Unexpected argument: " + option +
                    ". Use PQ --help for usage."
                );
        }

        return;
    }

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

/**
 * @brief get the requested output format
 *
 * @return CommandLineFormat
 */
CommandLineFormat CommandLineArgs::getFormat() const { return _format; }

/**
 * @brief get the requested validation scope
 *
 * @return ValidationScope
 */
ValidationScope CommandLineArgs::getValidationScope() const
{
    return _validationScope;
}
