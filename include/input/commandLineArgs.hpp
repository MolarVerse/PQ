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

#ifndef _COMMAND_LINE_ARGS_HPP_

#define _COMMAND_LINE_ARGS_HPP_

#include <string>
#include <vector>

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

  public:
    CommandLineArgs(const int argc, const std::vector<std::string> &argv) : _argc(argc), _argv(argv){};

    void detectFlags();

    std::string getInputFileName() const { return _inputFileName; }
};

#endif   // _COMMAND_LINE_ARGS_HPP_