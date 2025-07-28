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

#include "ringPolymerRestartFileReader.hpp"

#include <cstddef>       // for size_t
#include <format>        // for format
#include <fstream>       // IWYU pragma: keep
#include <memory>        // for __shared_ptr_access, shared_ptr
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "atom.hpp"                  // for Atom
#include "exceptions.hpp"            // for RingPolymerRestartFileException
#include "fileSettings.hpp"          // for FileSettings
#include "ringPolymerEngine.hpp"     // for RingPolymerEngine
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "stringUtilities.hpp"       // for removeComments, splitString

using input::ringPolymer::RingPolymerRestartFileReader;
using namespace engine;
using namespace settings;
using namespace customException;
using namespace utilities;

/**
 * @brief Construct a new Ring Polymer Restart File Reader:: Ring Polymer
 * Restart File Reader object
 *
 * @param fileName
 * @param engine
 */
RingPolymerRestartFileReader::RingPolymerRestartFileReader(
    const std::string &fileName,
    RingPolymerEngine &engine
)
    : _fileName(fileName), _fp(fileName), _engine(engine)
{
}

/**
 * @brief Reads a .rpmd.rst file sets the ring polymer beads in the engine
 *
 */
void RingPolymerRestartFileReader::read()
{
    std::string              line;
    std::vector<std::string> lineElements;
    int                      lineNumber = 0;

    const auto numberOfBeads = RingPolymerSettings::getNumberOfBeads();

    for (size_t i = 0; i < numberOfBeads; ++i)
    {
        for (auto &atom : _engine.getRingPolymerBeads()[i].getAtoms())
        {
            do {
                if (!getline(_fp, line))
                    throw RingPolymerRestartFileException(
                        "Error reading ring polymer restart file"
                    );

                line         = removeComments(line, "#");
                lineElements = splitString(line);
                ++lineNumber;

            } while (lineElements.empty());

            if ((lineElements.size() != 21) && (lineElements.size() != 12))
                throw RstFileException(
                    std::format(
                        "Error in line {}: Atom section must have 12 or 21 "
                        "elements",
                        lineNumber
                    )
                );

            atom->setPosition(
                {stod(lineElements[3]),
                 stod(lineElements[4]),
                 stod(lineElements[5])}
            );

            atom->setVelocity(
                {stod(lineElements[6]),
                 stod(lineElements[7]),
                 stod(lineElements[8])}
            );

            atom->setForce(
                {stod(lineElements[9]),
                 stod(lineElements[10]),
                 stod(lineElements[11])}
            );
        }
    }
}

/**
 * @brief wrapper function to construct a RingPolymerRestartFileReader object
 * and call the read function
 *
 * @param engine
 */
void input::ringPolymer::readRingPolymerRestartFile(RingPolymerEngine &engine)
{
    const auto filename = FileSettings::getRingPolymerStartFileName();

    RingPolymerRestartFileReader reader(filename, engine);

    reader.read();
}