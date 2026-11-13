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

#include "orcaRunner.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <mstd/file.hpp>
#include <sstream>

#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "hybridConfigurator.hpp"
#include "hybridSettings.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"
#include "stringUtilities.hpp"

namespace QM
{
    /**
     * @brief writes the coords file in ORCA format
     *
     * @param simulationBox
     */
    void OrcaRunner::writeCoordsFile(const molsys::SimulationBox &simulationBox)
    {
        auto file = std::ofstream("orca_coord.xyz");
        file << simulationBox.getNumberOfQMAtoms() << "\n\n";
        for (const auto &atom : simulationBox.getQMAtoms())
        {
            const auto pos = atom->getPosition();
            file << std::format(
                "{} {:.12f} {:.12f} {:.12f}\n",
                atom->getName(),
                pos[0],
                pos[1],
                pos[2]
            );
        }
        file.close();
    }

    /**
     * @brief writes the point charge file in ORCA format
     *
     * @param simulationBox
     */
    void OrcaRunner::writePointChargeFile(
        const molsys::SimulationBox &simulationBox
    )
    {
        _usePointCharges = false;
        auto   body      = std::ostringstream();
        size_t pcCount   = 0;

        using enum molsys::HybridZone;

        for (const auto &mol : simulationBox.getInactiveMolecules())
        {
            const auto zone = mol.getHybridZone();

            if (zone == SMOOTHING || zone == POINT_CHARGE)
            {
                for (const auto &atom : mol.getAtoms())
                {
                    const auto pos = atom->getPosition();
                    body << std::format(
                        "{:.12f} {:.12f} {:.12f} {:.12f}\n",
                        atom->getPartialCharge(),
                        pos[0],
                        pos[1],
                        pos[2]
                    );
                    ++pcCount;
                }
            }
        }
        const auto pcFileName = FileSettings::getPointChargeFileName();

        if (pcCount == 0)
        {
            std::filesystem::remove(pcFileName);
            return;
        }

        auto pcFile = std::ofstream(pcFileName);
        pcFile << pcCount << '\n' << body.str();
        pcFile.close();
        _usePointCharges = true;
    }

    /**
     * @brief executes ORCA as the external qm program
     *
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     */
    void OrcaRunner::execute(const molsys::SimulationBox &simulationBox)
    {
        using enum SmoothingMethod;

        const auto scriptFile = _getExternalQMSettings().getQMScript();

        if (!mstd::File(scriptFile).exists())
        {
            throw exc::InputFileException(
                std::format(
                    "ORCA script file \"{}\" does not exist.",
                    scriptFile
                )
            );
        }

        auto charge = simulationBox.calcActiveMolCharge();
        auto molChangedZone =
            configurator::HybridConfigurator::getMoleculeChangedZone();

        // TODO(ape): https://github.com/MolarVerse/PQ/issues/200
        if (settings::HybridSettings::getSmoothingMethod() == EXACT)
            molChangedZone = true;

        const auto reset           = molChangedZone || _isFirstExecution;
        const auto usePointCharges = _usePointCharges ? 1 : 0;

        const auto command = std::format(
            "{} {} {} {} {} {}",
            utilities::shellQuote(scriptFile),
            charge,
            reset ? 1 : 0,
            usePointCharges,
            utilities::shellQuote(FileSettings::getOrcaFileName()),
            utilities::shellQuote(FileSettings::getPointChargeFileName())
        );
        _executeCommand(command, "ORCA");

        _isFirstExecution = false;
        _usePointCharges  = false;
    }
}   // namespace QM
