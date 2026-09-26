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

#include "pyscfRunner.hpp"

#include <format>
#include <fstream>
#include <mstd/file.hpp>
#include <string>

#include "exceptions.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"
#include "stringUtilities.hpp"

using QM::PySCFRunner;
using namespace molsys;
using namespace settings;
using namespace exc;
using namespace utilities;

/**
 * @brief writes the coords file in order to run the external qm program
 *
 * @param simulationBox Simulation box containing molecules and atoms.
 */
void PySCFRunner::writeCoordsFile(SimulationBox &simulationBox)
{
    const std::string fileName = "coords.xyz";
    std::ofstream     coordsFile(fileName);

    coordsFile << simulationBox.getNumberOfQMAtoms() << "\n\n";

    for (const auto &atom : simulationBox.getQMAtoms())
    {
        coordsFile << std::format(
            "{:5s}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
            atom->getName(),
            atom->getPosition()[0],
            atom->getPosition()[1],
            atom->getPosition()[2]
        );
    }

    coordsFile.close();
}

/**
 * @brief executes the qm script of the external program
 *
 */
void PySCFRunner::execute(SimulationBox & /*simBox*/)
{
    const auto scriptFileName = resolveScriptPath(QMSettings::getQMScript());

    if (!mstd::File(scriptFileName).exists())
    {
        throw InputFileException(
            std::format(
                "PySCF script file \"{}\" does not exist.",
                scriptFileName
            )
        );
    }

    const auto command = std::format(
        "python {} > {}",
        shellQuote(scriptFileName),
        shellQuote("pyscf.out")
    );

    executeCommand(command, "PySCF");
}
