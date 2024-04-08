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

#include "output.hpp"

#include "exceptions.hpp"   // for InputFileException, customException

#include <filesystem>   // for create_directory
#include <fstream>      // for ifstream, ofstream, std

using namespace std;
using namespace customException;
using namespace output;

/**
 * @brief Sets the filename of the output file
 *
 * @param filename
 *
 * @throw InputFileException if filename is empty
 * @throw InputFileException if file already exists
 */
void Output::setFilename(const string_view &filename)
{
    _fileName = filename;

    if (_fileName.empty())
        throw InputFileException("Filename cannot be empty");

    if (const ifstream fp(_fileName.c_str()); fp.good())
        throw InputFileException("File already exists - filename = " + string(_fileName));

    openFile();
}

/**
 * @brief Opens the output file
 *
 * @throw InputFileException if file cannot be opened
 *
 */
void Output::openFile()
{
    _fp.open(_fileName);

    if (!_fp.is_open())
        throw InputFileException("Could not open file - filename = " + _fileName);
}