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

#include "topologySection.hpp"

#include "stringUtilities.hpp"   // for removeComments, splitString, toLowerCopy

#include <fstream>   // for getline

using namespace input::topology;

/**
 * @brief general process function for topology sections
 *
 * @details Reads the topology file line by line and calls the processSection function for each line until the "end" keyword is
 * found. At the end of the section the endedNormally function is called, which checks if the "end" keyword was found.
 *
 * @param line
 * @param engine
 */
void TopologySection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    std::string line;
    auto        endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
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