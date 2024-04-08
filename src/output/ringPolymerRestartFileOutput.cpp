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

#include "ringPolymerRestartFileOutput.hpp"

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for operator<<

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<
#include <vector>    // for vector

using output::RingPolymerRestartFileOutput;

/**
 * @brief Write the restart file for all beads
 *
 * @param simBox
 * @param step
 */
void RingPolymerRestartFileOutput::write(std::vector<simulationBox::SimulationBox> &beads, const size_t step)
{
    _fp.close();

    _fp.open(_fileName);

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);
                _fp << std::format("{:>5}\t", j + 1);
                _fp << std::format("{:>5}\t", molecule.getMoltype());

                _fp << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t",
                                   molecule.getAtomPosition(j)[0],
                                   molecule.getAtomPosition(j)[1],
                                   molecule.getAtomPosition(j)[2]);

                _fp << std::format("{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                                   molecule.getAtomVelocity(j)[0],
                                   molecule.getAtomVelocity(j)[1],
                                   molecule.getAtomVelocity(j)[2]);

                _fp << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}",
                                   molecule.getAtomForce(j)[0],
                                   molecule.getAtomForce(j)[1],
                                   molecule.getAtomForce(j)[2]);

                _fp << '\n' << std::flush;
            }
        }
}