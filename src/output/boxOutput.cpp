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

#include <cstddef>
#include <format>
#include <ostream>

#include "box.hpp"
#include "units.hpp"

namespace out
{
    /**
     * @brief Write the lattice parameters a, b, c, alpha, beta, gamma to file
     *
     * @param step
     * @param box
     */
    void BoxFileOutput::write(size_t step, const molsys::Box& box)
    {
        _getFilePointer() << std::format("{:<5}\t", step);

        const auto& boxDimensions =
            box.get_boxDimensions().numerical_value_in(angstrom);

        _getFilePointer() << std::format(
            "{:15.8f}\t{:15.8f}\t{:15.8f}\t",
            boxDimensions[0],
            boxDimensions[1],
            boxDimensions[2]
        );

        const auto& boxAngles = box.getBoxAngles().numerical_value_in(deg);

        _getFilePointer() << std::format(
            "{:15.8f}\t{:15.8f}\t{:15.8f}\n",
            boxAngles[0],
            boxAngles[1],
            boxAngles[2]
        );

        _getFilePointer() << std::flush;
    }
}   // namespace out
