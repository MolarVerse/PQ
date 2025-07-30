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

#include "dftbplusRunner.hpp"

#include <algorithm>    // for std::ranges:find
#include <cstddef>      // for size_t
#include <cstdlib>      // for system
#include <format>       // for format
#include <fstream>      // for ofstream
#include <functional>   // for identity
#include <iterator>     // for std::ranges::distance
#include <set>          // for set
#include <string>       // for string
#include <vector>       // for vector

#include "atom.hpp"              // for Atom
#include "exceptions.hpp"        // for InputFileException
#include "fileSettings.hpp"      // for FileSettings
#include "physicalData.hpp"      // for PhysicalData
#include "qmSettings.hpp"        // for QMSettings
#include "settings.hpp"          // for Settings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for fileExists
#include "vector3d.hpp"          // for Vec3D

using QM::DFTBPlusRunner;
using enum QM::Periodicity;

using namespace simulationBox;
using namespace physicalData;
using namespace customException;
using namespace settings;
using namespace constants;
using namespace utilities;
using namespace linearAlgebra;

/**
 * @brief writes the coords file in order to run the external qm program
 *
 * @param box
 */
void DFTBPlusRunner::writeCoordsFile(SimulationBox &box)
{
    using std::ranges::distance;
    using std::ranges::find;

    const std::string fileName = "coords";
    std::ofstream     coordsFile(fileName);

    coordsFile << box.getNumberOfQMAtoms();
    coordsFile << "  " << (_periodicity == NON_PERIODIC ? 'C' : 'S') << '\n';

    const auto uniqueAtomNames = box.getUniqueQMAtomNames();

    for (const auto &atomName : uniqueAtomNames) coordsFile << atomName << "  ";
    coordsFile << "\n";

    size_t atomIndex = 1;
    for (const auto &atom : box.getQMAtoms())
    {
        const auto iter   = find(uniqueAtomNames, atom->getName());
        const auto atomId = distance(uniqueAtomNames.begin(), iter) + 1;

        coordsFile << std::format(
            "{:5d} {:5d}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            atomIndex,
            atomId,
            atom->getPosition()[0],
            atom->getPosition()[1],
            atom->getPosition()[2]
        );
        ++atomIndex;
    }

    if (_periodicity != NON_PERIODIC)
    {
        auto             boxMatrix  = box.getBox().getBoxMatrix();
        constexpr double vacuumSize = 200.0;   // Large vacuum spacing in Å

        switch (_periodicity)
        {
            case X:
                // Periodic in X only, set Y and Z to large values
                boxMatrix[1][1] = vacuumSize;              // Y dimension
                boxMatrix[2][2] = vacuumSize;              // Z dimension
                boxMatrix[0][1] = boxMatrix[0][2] = 0.0;   // Clear cross terms
                boxMatrix[1][0] = boxMatrix[1][2] = 0.0;
                boxMatrix[2][0] = boxMatrix[2][1] = 0.0;
                break;
            case Y:
                // Periodic in Y only, set X and Z to large values
                boxMatrix[0][0] = vacuumSize;              // X dimension
                boxMatrix[2][2] = vacuumSize;              // Z dimension
                boxMatrix[0][1] = boxMatrix[0][2] = 0.0;   // Clear cross terms
                boxMatrix[1][0] = boxMatrix[1][2] = 0.0;
                boxMatrix[2][0] = boxMatrix[2][1] = 0.0;
                break;
            case Z:
                // Periodic in Z only, set X and Y to large values
                boxMatrix[0][0] = vacuumSize;              // X dimension
                boxMatrix[1][1] = vacuumSize;              // Y dimension
                boxMatrix[0][1] = boxMatrix[0][2] = 0.0;   // Clear cross terms
                boxMatrix[1][0] = boxMatrix[1][2] = 0.0;
                boxMatrix[2][0] = boxMatrix[2][1] = 0.0;
                break;
            case XY:
                // Periodic in XY, set Z to large value
                boxMatrix[2][2] = vacuumSize;   // Z dimension
                boxMatrix[0][2] = boxMatrix[2][0] =
                    0.0;   // Clear XZ cross terms
                boxMatrix[1][2] = boxMatrix[2][1] =
                    0.0;   // Clear YZ cross terms
                break;
            case XZ:
                // Periodic in XZ, set Y to large value
                boxMatrix[1][1] = vacuumSize;   // Y dimension
                boxMatrix[0][1] = boxMatrix[1][0] =
                    0.0;   // Clear XY cross terms
                boxMatrix[1][2] = boxMatrix[2][1] =
                    0.0;   // Clear YZ cross terms
                break;
            case YZ:
                // Periodic in YZ, set X to large value
                boxMatrix[0][0] = vacuumSize;   // X dimension
                boxMatrix[0][1] = boxMatrix[1][0] =
                    0.0;   // Clear XY cross terms
                boxMatrix[0][2] = boxMatrix[2][0] =
                    0.0;   // Clear XZ cross terms
                break;
            default: break;
        }

        // coordinate origin
        coordsFile << std::format(
            "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            "",
            0.0,
            0.0,
            0.0
        );

        // lattice vector a
        coordsFile << std::format(
            "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            "",
            boxMatrix[0][0],
            boxMatrix[1][0],
            boxMatrix[2][0]
        );

        // lattice vector b
        coordsFile << std::format(
            "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            "",
            boxMatrix[0][1],
            boxMatrix[1][1],
            boxMatrix[2][1]
        );

        // lattice vector c
        coordsFile << std::format(
            "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            "",
            boxMatrix[0][2],
            boxMatrix[1][2],
            boxMatrix[2][2]
        );
    }

    coordsFile.close();
}

/**
 * @brief executes the qm script of the external program
 *
 */
void DFTBPlusRunner::execute()
{
    const auto scriptFile = _scriptPath + QMSettings::getQMScript();

    if (!fileExists(scriptFile))
        throw InputFileException(
            std::format("DFTB+ script file \"{}\" does not exist.", scriptFile)
        );

    const auto reuseCharges = _isFirstExecution ? 1 : 0;

    const auto command = std::format(
        "{} 0 {} 0 0 0 {}",
        scriptFile,
        reuseCharges,
        FileSettings::getDFTBFileName()
    );
    ::system(command.c_str());

    _isFirstExecution = false;
}

/**
 * @brief reads the stress tensor and adds it to the physical data
 *
 * @param box
 * @param data
 */
void DFTBPlusRunner::readStressTensor(Box &box, PhysicalData &data)
{
    if (_periodicity == NON_PERIODIC)
        return;

    const std::string stressFileName = "stress_tensor";

    std::ifstream stressFile(stressFileName);

    if (!stressFile.is_open())
        throw QMRunnerException(
            std::format(
                "Cannot open {} stress tensor \"{}\"",
                string(QMSettings::getQMMethod()),
                stressFileName
            )
        );

    StaticMatrix3x3<double> stress;

    stressFile >> stress[0][0] >> stress[0][1] >> stress[0][2];
    stressFile >> stress[1][0] >> stress[1][1] >> stress[1][2];
    stressFile >> stress[2][0] >> stress[2][1] >> stress[2][2];

    const auto conversion = _HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3_;
    stress                = stress * conversion;
    const auto virial     = stress * box.getVolume();

    data.setStressTensor(stress);
    data.addVirial(virial);

    stressFile.close();

    ::system(std::format("rm -f {}", stressFileName).c_str());
}