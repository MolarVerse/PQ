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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mstd/file.hpp>
#include <set>
#include <string>

#include "box.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "hybridConfigurator.hpp"
#include "hybridSettings.hpp"
#include "physicalData.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"
#include "stringUtilities.hpp"
#include "units.hpp"

namespace QM
{

    /**
     * @brief writes the coords file in order to run the external qm program
     *
     * @param simulationBox
     */
    void DFTBPlusRunner::_writeCoordsFile(
        const molsys::SimulationBox &simulationBox
    )
    {
        using std::ranges::distance;
        using std::ranges::find;

        const std::string fileName = "coords";
        std::ofstream     coordsFile(fileName);

        coordsFile << simulationBox.getNumberOfActiveAtoms();
        coordsFile << "  "
                   << (_getPeriodicity() == molsys::Periodicity::NON_PERIODIC
                           ? 'C'
                           : 'S')
                   << '\n';

        const auto uniqueAtomNames = simulationBox.getUniqueActiveAtomNames();

        for (const auto &atomName : uniqueAtomNames)
            coordsFile << atomName << "  ";
        coordsFile << "\n";

        size_t atomIndex = 1;
        for (const auto &atom : simulationBox.getActiveAtoms())
        {
            const auto iter   = find(uniqueAtomNames, atom->getName());
            const auto atomId = distance(uniqueAtomNames.begin(), iter) + 1;

            coordsFile << std::format(
                "{:5d} {:5d}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                atomIndex,
                atomId,
                atom->getPosition()[0].numerical_value_in(angstrom),
                atom->getPosition()[1].numerical_value_in(angstrom),
                atom->getPosition()[2].numerical_value_in(angstrom)
            );
            ++atomIndex;
        }

        if (_getPeriodicity() != molsys::Periodicity::NON_PERIODIC)
        {
            const auto boxMatrix =
                simulationBox.getBox().getBoxMatrix(_getPeriodicity());

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
                boxMatrix[0][0].numerical_value_in(angstrom),
                boxMatrix[1][0].numerical_value_in(angstrom),
                boxMatrix[2][0].numerical_value_in(angstrom)
            );

            coordsFile << std::format(
                "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                "",
                boxMatrix[0][1].numerical_value_in(angstrom),
                boxMatrix[1][1].numerical_value_in(angstrom),
                boxMatrix[2][1].numerical_value_in(angstrom)
            );

            coordsFile << std::format(
                "{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                "",
                boxMatrix[0][2].numerical_value_in(angstrom),
                boxMatrix[1][2].numerical_value_in(angstrom),
                boxMatrix[2][2].numerical_value_in(angstrom)
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
     * @param pointChargeFileName Path to the file where point charges will be
     * written.
     *
     * @throws exc::QMRunnerException if the point charges file cannot be
     * opened, written to, or if it contains invalid values.
     */
    void DFTBPlusRunner::_writePointChargeFile(
        const molsys::SimulationBox &simulationBox,
        const std::string           &pointChargeFileName
    )
    {
        std::ofstream pcFile(pointChargeFileName);

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
                        atom->getPosition()[0].numerical_value_in(angstrom),
                        atom->getPosition()[1].numerical_value_in(angstrom),
                        atom->getPosition()[2].numerical_value_in(angstrom),
                        atom->getPartialCharge().numerical_value_in(
                            elementary_charge
                        )
                    );
                    _usePointCharges = true;
                }
            }
        }
        pcFile.close();

        if (!_usePointCharges)
            std::filesystem::remove(pointChargeFileName);
    }

    /**
     * @brief executes the qm script of the external program
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     * @param fileSettings Object containing file paths and settings required
     * for the execution of the external QM program.
     * @param hybridSettings The hybrid settings of the system, which may
     * influence the execution of the QM script, particularly in the context of
     * hybrid QM/MM simulations.
     *
     */
    void DFTBPlusRunner::execute(
        const molsys::SimulationBox &simulationBox,
        const FileSettings          &fileSettings,
        const HybridSettings        &hybridSettings
    )
    {
        const auto scriptFile = _getExternalQMSettings().getQMScript();

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

        // TODO(ape): https://github.com/MolarVerse/PQ/issues/200
        if (hybridSettings.getSmoothingMethod() == SmoothingMethod::EXACT)
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
            utilities::shellQuote(fileSettings.get_dftbFile()),
            utilities::shellQuote(fileSettings.get_pointChargeFile())
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
     * @param stressFileName Path to the file containing the stress tensor.
     * @throws exc::QMRunnerException if the stress tensor file cannot be
     * opened, if it is incomplete, or if it contains invalid values.
     */
    void DFTBPlusRunner::_readStressTensor(
        const molsys::Box          &box,
        physicalData::PhysicalData &physicalData,
        const std::string          &stressFileName
    )
    {
        std::ifstream stressFile(stressFileName);

        if (!stressFile.is_open())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Cannot open {} stress tensor \"{}\"",
                    QMMethodMeta::toString(_getQMSettings().getQMMethod()),
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
                    QMMethodMeta::toString(_getQMSettings().getQMMethod()),
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
                                _getQMSettings().getQMMethod()
                            ),
                            stressFileName
                        )
                    );
                }
            }
        }

        const Stress stress_ = stress * hartree / bohr3 * avogadro_constant;
        const auto   virial  = stress_ * box.getVolume();

        physicalData.set_stressTensor(stress_);
        physicalData.addVirial(virial);

        stressFile.close();
    }

}   // namespace QM
