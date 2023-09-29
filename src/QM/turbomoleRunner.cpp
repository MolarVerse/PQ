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

#include "turbomoleRunner.hpp"

#include "atom.hpp"              // for Atom
#include "exceptions.hpp"        // for InputFileException
#include "qmSettings.hpp"        // for QMSettings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for fileExists
#include "vector3d.hpp"          // for Vec3D

#include <cstddef>      // for size_t
#include <cstdlib>      // for system
#include <format>       // for format
#include <fstream>      // for ofstream
#include <functional>   // for identity
#include <ranges>       // for borrowed_iterator_t, __distance_fn
#include <string>       // for string
#include <vector>       // for vector

using QM::TurbomoleRunner;

/**
 * @brief writes the coords file in turbomole format
 *
 * @param box
 */
void DFTBPlusRunner::writeCoordsFile(simulationBox::SimulationBox &box)
{
    const std::string fileName = "coords";
    std::ofstream     coordsFile(fileName);

    coordsFile << "$coord\n";
    for (size_t i = 0, numberOfAtoms = box.getNumberOfQMAtoms(); i < numberOfAtoms; ++i)
    {
        const auto &atom = box.getQMAtom(i);

        coordsFile << std::format("\t{:16.12f}\t{:16.12f}\t{:16.12f}\t{}\n",
                                  atom.getPosition()[0],
                                  atom.getPosition()[1],
                                  atom.getPosition()[2],
                                  atom.getName());
    }

    coordsFile << "$end\n";

    coordsFile.close();
}

/**
 * @brief TODO: not yet implemented
 *
 */
void DFTBPlusRunner::execute()
{
    const auto scriptFileName = _scriptPath + settings::QMSettings::getQMScript();

    if (!utilities::fileExists(scriptFileName))
        throw customException::InputFileException(std::format("Turbomole script file \"{}\" does not exist.", scriptFileName));

    const auto reuseCharges = _isFirstExecution ? 1 : 0;

    const auto command = std::format("{} 0 {} 0 0 0", scriptFileName, reuseCharges);
    ::system(command.c_str());

    _isFirstExecution = false;
}