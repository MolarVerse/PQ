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

#include "rstFileOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for operator<<

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<
#include <vector>    // for vector

using namespace output;

/**
 * @brief Write the restart file
 *
 * @param simBox
 * @param step
 */
void RstFileOutput::write(simulationBox::SimulationBox &simBox, const size_t step)
{
    _fp.close();

    _fp.open(_fileName);

    _fp << "Step " << step << '\n';

    _fp << "Box   " << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << std::format("{:<5}\t", molecule.getAtomName(i));
            _fp << std::format("{:<5}\t", i + 1);
            _fp << std::format("{:<5}\t", molecule.getMoltype());

            _fp << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t",
                               molecule.getAtomPosition(i)[0],
                               molecule.getAtomPosition(i)[1],
                               molecule.getAtomPosition(i)[2]);

            _fp << std::format("{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                               molecule.getAtomVelocity(i)[0],
                               molecule.getAtomVelocity(i)[1],
                               molecule.getAtomVelocity(i)[2]);

            _fp << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}",
                               molecule.getAtomForce(i)[0],
                               molecule.getAtomForce(i)[1],
                               molecule.getAtomForce(i)[2]);

            _fp << '\n' << std::flush;
        }
    }
}