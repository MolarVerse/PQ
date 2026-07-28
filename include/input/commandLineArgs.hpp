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

#ifndef _COMMAND_LINE_ARGS_HPP_

#define _COMMAND_LINE_ARGS_HPP_

#include <string>
#include <vector>

enum class CommandLineAction
{
    RUN,
    HELP,
    VERSION,
    CAPABILITIES,
    VALIDATE
};

enum class CommandLineFormat
{
    TEXT,
    JSON
};

enum class ValidationScope
{
    INSTALLED,
    PORTABLE
};

/**
 * @class CommandLineArgs
 *
 * @brief Handles the command line arguments.
 *
 */
class CommandLineArgs
{
   private:
    int                      _argc;
    std::vector<std::string> _argv;
    std::string              _inputFileName;
    CommandLineAction        _action          = CommandLineAction::RUN;
    CommandLineFormat        _format          = CommandLineFormat::TEXT;
    ValidationScope          _validationScope = ValidationScope::INSTALLED;

   public:
    CommandLineArgs(const int argc, const std::vector<std::string> &argv);

    void parse();

    std::string       getInputFileName() const;
    CommandLineAction getAction() const;
    CommandLineFormat getFormat() const;
    ValidationScope   getValidationScope() const;
};

#endif   // _COMMAND_LINE_ARGS_HPP_
