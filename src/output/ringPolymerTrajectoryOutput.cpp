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

#include "ringPolymerTrajectoryOutput.hpp"

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for Vec3D, operator<<

#include <algorithm>    // for __for_each_fn, for_each
#include <format>       // for format
#include <functional>   // for identity
#include <ostream>      // for basic_ostream, ofstream, operator<<
#include <stddef.h>     // for size_t
#include <string>       // for operator<<, char_traits

using output::RingPolymerTrajectoryOutput;

/**
 * @brief write the header of the beads trajectory file
 *
 * @details number of atoms is multiplied by the number of beads - box dimensions and angles are the same for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeHeader(const simulationBox::SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms() * settings::RingPolymerSettings::getNumberOfBeads() << "  ";
    _fp << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << '\n';
}

/**
 * @brief write the xyz file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeXyz(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(j)[0]);
                _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(j)[1]);
                _fp << std::format("{:15.8f}\n", molecule.getAtomPosition(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the velocity file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeVelocities(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(j)[0]);
                _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(j)[1]);
                _fp << std::format("{:20.8e}\n", molecule.getAtomVelocity(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the force file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeForces(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);

    auto totalForce = 0.0;
    std::ranges::for_each(beads, [&totalForce](auto &bead) { totalForce += bead.calculateTotalForce(); });

    _fp << std::format("# Total force = {:.5e} kcal/mol/Angstrom\n", totalForce);

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\t", molecule.getAtomForce(j)[0]);
                _fp << std::format("{:15.8f}\t", molecule.getAtomForce(j)[1]);
                _fp << std::format("{:15.8f}\n", molecule.getAtomForce(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the charge file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeCharges(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\n", molecule.getPartialCharge(j));

                _fp << std::flush;
            }
}