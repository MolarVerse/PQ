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

#include "boxOutput.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <ostream>   // for ofstream, basic_ostream, operator<<
#include <string>    // for operator<<
#include <vector>    // for vector

#include "box.hpp"        // for SimulationBox
#include "vector3d.hpp"   // for Vec3D

using output::BoxFileOutput;

/**
 * @brief Write the lattice parameters a, b, c, alpha, beta, gamma to file
 *
 * @param box
 */
void BoxFileOutput::write(const size_t step, const simulationBox::Box &box)
{
    _fp << std::format("{:<5}\t", step);

    _fp << std::format(
        "{:15.8f}\t{:15.8f}\t{:15.8f}\t",
        box.getBoxDimensions()[0],
        box.getBoxDimensions()[1],
        box.getBoxDimensions()[2]
    );

    _fp << std::format(
        "{:15.8f}\t{:15.8f}\t{:15.8f}\n",
        box.getBoxAngles()[0],
        box.getBoxAngles()[1],
        box.getBoxAngles()[2]
    );

    _fp << std::flush;
}