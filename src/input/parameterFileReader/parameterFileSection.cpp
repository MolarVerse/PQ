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

#include "parameterFileSection.hpp"

#include <fstream>   // for getline

#include "exceptions.hpp"        // for ParameterFileException
#include "stringUtilities.hpp"   // for removeComments, splitString, toLowerCopy

using namespace input::parameterFile;
using namespace utilities;
using namespace customException;
using namespace engine;

/**
 * @brief reads a general parameter file section
 *
 * @details Calls processHeader at the beginning of each section and
 * processSection for each line in the section. If the "end" keyword is found,
 * the section is ended normally.
 *
 * @param line
 * @param engine
 */
void ParameterFileSection::process(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    processHeader(lineElements, engine);

    std::string line;
    auto        endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (toLowerCopy(lineElements[0]) == "end")
        {
            ++_lineNumber;
            endedNormal = true;
            break;
        }

        processSection(lineElements, engine);

        ++_lineNumber;
    }

    endedNormally(endedNormal);
}

/**
 * @brief check if section ended normally
 *
 * @param endedNormally
 *
 * @throw ParameterFileException if section did not end
 * normally
 */
void ParameterFileSection::endedNormally(const bool endedNormally)
{
    if (!endedNormally)
        throw ParameterFileException(
            "Parameter file " + keyword() + " section ended abnormally!"
        );
}

/**
 * @brief set line number of section
 *
 * @param lineNumber
 */
void ParameterFileSection::setLineNumber(const int lineNumber)
{
    _lineNumber = lineNumber;
}

/**
 * @brief set file pointer
 *
 * @param fp
 */
void ParameterFileSection::setFp(std::ifstream *fp) { _fp = fp; }

/**
 * @brief get line number of section
 *
 * @return int
 */
int ParameterFileSection::getLineNumber() const { return _lineNumber; }