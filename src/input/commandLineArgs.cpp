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

#include <string_view>   // for string_view

/**
 * @brief Detects flags in the command line arguments. First argument is the input file name.
 *
 * @throw UserInputException if a flag is detected (not yet implemented)
 * @throw UserInputException if no input file is specified
 */
void CommandLineArgs::detectFlags()
{
    for (const auto &arg : _argv)
        if ('-' == arg[0])
            throw customException::UserInputException("Invalid flag: " + arg + " Flags are not yet implemented.");

    if (_argc < 2)
        throw customException::UserInputException("No input file specified. Usage: PQ <input_file>");

    _inputFileName = _argv[1];
}