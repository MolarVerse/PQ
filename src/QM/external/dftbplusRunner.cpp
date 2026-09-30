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
#include <cmath>        // for isfinite
#include <cstddef>      // for size_t
#include <filesystem>   // for remove
#include <format>       // for format
#include <fstream>      // for ofstream
#include <iterator>     // for std::ranges::distance
#include <mstd/file.hpp>
#include <set>      // for set
#include <string>   // for string

#include "box.hpp"   // for molsys::Periodicity
#include "constants.hpp"
#include "exceptions.hpp"           // for InputFileException
#include "fileSettings.hpp"         // for FileSettings
#include "hybridConfigurator.hpp"   // for HybridConfigurator
#include "hybridSettings.hpp"       // for SmoothingMethod
#include "physicalData.hpp"         // for PhysicalData
#include "qmSettings.hpp"           // for QMSettings
#include "simulationBox.hpp"        // for SimulationBox
#include "stringUtilities.hpp"      // for fileExists

namespace QM
{

    /**
     * @brief writes the coords file in order to run the external qm program
     *
     * @param simulationBox
     */
    void DFTBPlusRunner::writeCoordsFile(molsys::SimulationBox &simulationBox)
    {
        using std::ranges::distance;
        using std::ranges::find;

        const std::string fileName = "coords";
        std::ofstream     coordsFile(fileName);

        coordsFile << simulationBox.getNumberOfQMAtoms();
        coordsFile << "  "
                   << (_periodicity == molsys::Periodicity::NON_PERIODIC ? 'C'
                                                                         : 'S')
                   << '\n';

        const auto uniqueAtomNames = simulationBox.getUniqueQMAtomNames();

        for (const auto &atomName : uniqueAtomNames)
            coordsFile << atomName << "  ";
        coordsFile << "\n";

        size_t atomIndex = 1;
        for (const auto &atom : simulationBox.getQMAtoms())
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

        if (_periodicity != molsys::Periodicity::NON_PERIODIC)
        {
            const auto boxMatrix =
                simulationBox.getBox().getBoxMatrix(_periodicity);

            // coordinate origin
            coordsFile << std::format(
                "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                "",
                0.0,
                0.0,
                0.0
            );

            coordsFile << std::format(
                "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                "",
                boxMatrix[0][0],
                boxMatrix[1][0],
                boxMatrix[2][0]
            );

            coordsFile << std::format(
                "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                "",
                boxMatrix[0][1],
                boxMatrix[1][1],
                boxMatrix[2][1]
            );

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
     * @brief Writes a file containing point charges for hybrid simulations.
     *
     * This function creates the pointcharges file listing the positions and
     * partial charges of all atoms in inactive molecules assigned to the
     * SMOOTHING or POINT_CHARGE hybrid zones. The file is used for QM/QM and
     * QM/MM coupling in DFTB+ calculations.
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     */
    void DFTBPlusRunner::writePointChargeFile(
        molsys::SimulationBox &simulationBox
    )
    {
        const std::string fileName =
            settings::FileSettings::getPointChargeFileName();
        std::ofstream pcFile(fileName);

        using enum molsys::HybridZone;

        for (const auto &mol : simulationBox.getInactiveMolecules())
        {
            const auto zone = mol.getHybridZone();

            if (zone == SMOOTHING || zone == POINT_CHARGE)
            {
                for (const auto &atom : mol.getAtoms())
                {
                    pcFile << std::format(
                        "{:16.12f}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                        atom->getPosition()[0],
                        atom->getPosition()[1],
                        atom->getPosition()[2],
                        atom->getPartialCharge()
                    );
                    _usePointCharges = true;
                }
            }
        }
        pcFile.close();

        if (!_usePointCharges)
            std::filesystem::remove(fileName);
    }

    /**
     * @brief executes the qm script of the external program
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     *
     */
    void DFTBPlusRunner::execute(molsys::SimulationBox &simulationBox)
    {
        const auto scriptFile =
            _resolveScriptPath(settings::QMSettings::getQMScript());

        if (!mstd::File(scriptFile).exists())
        {
            throw exc::InputFileException(
                std::format(
                    "DFTB+ script file \"{}\" does not exist.",
                    scriptFile
                )
            );
        }

        auto charge = simulationBox.calcActiveMolCharge();

        auto molChangedZone =
            configurator::HybridConfigurator::getMoleculeChangedZone();

        // TODO: https://github.com/MolarVerse/PQ/issues/200
        if (settings::HybridSettings::getSmoothingMethod() ==
            SmoothingMethod::EXACT)
            molChangedZone = true;

        const auto readChargesBin =
            !_isFirstExecution && !molChangedZone ? 1 : 0;
        const auto usePointCharges = _usePointCharges ? 1 : 0;

        const auto command = std::format(
            "{} {} {} {} {} {}",
            utilities::shellQuote(scriptFile),
            charge,
            readChargesBin,
            usePointCharges,
            utilities::shellQuote(settings::FileSettings::getDFTBFileName()),
            utilities::shellQuote(
                settings::FileSettings::getPointChargeFileName()
            )
        );
        _executeCommand(command, "DFTB+");

        // set for next execution
        _isFirstExecution = false;
        _usePointCharges  = false;
    }

    /**
     * @brief reads the stress tensor and adds it to the physical data
     *
     * @param box
     * @param physicalData Physical data object to which the stress tensor and
     * virial will be added.
     */
    void DFTBPlusRunner::readStressTensor(
        molsys::Box                &box,
        physicalData::PhysicalData &physicalData
    )
    {
        const auto stressFileName =
            settings::FileSettings::getStressTensorTempFileName();

        std::ifstream stressFile(stressFileName);

        if (!stressFile.is_open())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Cannot open {} stress tensor \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    stressFileName
                )
            );
        }

        linalg::StaticMatrix3x3<double> stress{0.0};

        if (!(stressFile >> stress[0][0] >> stress[0][1] >> stress[0][2] >>
              stress[1][0] >> stress[1][1] >> stress[1][2] >> stress[2][0] >>
              stress[2][1] >> stress[2][2]))
        {
            throw exc::QMRunnerException(
                std::format(
                    "Incomplete {} stress tensor \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    stressFileName
                )
            );
        }

        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t column = 0; column < 3; ++column)
            {
                if (!std::isfinite(stress[row][column]))
                {
                    throw exc::QMRunnerException(
                        std::format(
                            "Invalid value in {} stress tensor \"{}\"",
                            QMMethodMeta::toString(
                                settings::QMSettings::getQMMethod()
                            ),
                            stressFileName
                        )
                    );
                }
            }
        }

        const auto conversion = HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3;
        stress                = stress * conversion;
        const auto virial     = stress * box.getVolume();

        physicalData.setStressTensor(stress);
        physicalData.addVirial(virial);

        stressFile.close();
    }

}   // namespace QM
