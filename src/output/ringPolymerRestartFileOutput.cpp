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

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <sstream>   // for ostringstream
#include <string>    // for char_traits, operator<<
#include <vector>    // for vector

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for operator<<

using output::RingPolymerRestartFileOutput;

/**
 * @brief Write the restart file for all beads
 *
 * @param simBox
 * @param step
 */
void RingPolymerRestartFileOutput::write(
    std::vector<simulationBox::SimulationBox> &beads,
    const size_t                               step
)
{
    std::ostringstream buffer;

    _fp.close();
    _fp.open(_fileName);

    const auto nBeads = settings::RingPolymerSettings::getNumberOfBeads();

    for (size_t i = 0; i < nBeads; ++i)
        for (const auto &molecule : beads[i].getMolecules())
        {
            const size_t nAtoms = molecule.getNumberOfAtoms();
            for (size_t j = 0; j < nAtoms; ++j)
            {
                const auto atomName = molecule.getAtomName(j);
                const auto molType  = molecule.getMoltype();
                const auto x        = molecule.getAtomPosition(j)[0];
                const auto y        = molecule.getAtomPosition(j)[1];
                const auto z        = molecule.getAtomPosition(j)[2];
                const auto vx       = molecule.getAtomVelocity(j)[0];
                const auto vy       = molecule.getAtomVelocity(j)[1];
                const auto vz       = molecule.getAtomVelocity(j)[2];
                const auto fx       = molecule.getAtomForce(j)[0];
                const auto fy       = molecule.getAtomForce(j)[1];
                const auto fz       = molecule.getAtomForce(j)[2];

                buffer << std::format("{:>5}{}\t", atomName, i + 1);
                buffer << std::format("{:>5}\t", j + 1);
                buffer << std::format("{:>5}\t", molType);

                // clang-format off
                buffer << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t", x, y, z);
                buffer << std::format("{:19.8e}\t{:19.8e}\t{:19.8e}\t", vx, vy, vz);
                buffer << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}", fx, fy, fz);
                // clang-format on

                buffer << '\n';
            }
        }

    // Write the buffer to the file
    _fp << buffer.str();
    _fp << std::flush;
}