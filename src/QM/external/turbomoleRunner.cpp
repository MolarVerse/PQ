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

#include "turbomoleRunner.hpp"

#include <cstddef>      // for size_t
#include <cstdlib>      // for system
#include <format>       // for format
#include <fstream>      // for ofstream
#include <functional>   // for identity
#include <string>       // for string
#include <vector>       // for vector

#include "atom.hpp"              // for Atom
#include "constants.hpp"         // for constants
#include "exceptions.hpp"        // for InputFileException
#include "fileSettings.hpp"      // for FileSettings
#include "qmSettings.hpp"        // for QMSettings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for fileExists
#include "vector3d.hpp"          // for Vec3D

using QM::TurbomoleRunner;

using namespace simulationBox;
using namespace customException;
using namespace constants;
using namespace settings;
using namespace utilities;

/**
 * @brief writes the coords file in turbomole format
 *
 * @param box
 */
void TurbomoleRunner::writeCoordsFile(SimulationBox &box)
{
    const std::string fileName = "coord";
    std::ofstream     coordsFile(fileName);

    coordsFile << "$coord\n";

    for (const auto &atom : box.getQMAtoms())
    {
        const auto pos = atom->getPosition() * _ANGSTROM_TO_BOHR_;

        coordsFile << std::format(
            "   {:16.12f}   {:16.12f}   {:16.12f}   {}\n",
            pos[0],
            pos[1],
            pos[2],
            atom->getName()
        );
    }

    coordsFile << "$end\n";

    coordsFile.close();
}

/**
 * @brief executes the external qm program
 *
 */
void TurbomoleRunner::execute(SimulationBox &box)
{
    const auto scriptFile = _scriptPath + QMSettings::getQMScript();

    if (!fileExists(scriptFile))
        throw InputFileException(
            std::format(
                "Turbomole script file \"{}\" does not exist.",
                scriptFile
            )
        );

    const auto reuseCharges = _isFirstExecution ? 1 : 0;

    const auto command = std::format(
        "{} 0 {} 0 0 0 {}",
        scriptFile,
        reuseCharges,
        FileSettings::getPointChargeFileName()
    );
    ::system(command.c_str());

    _isFirstExecution = false;
}