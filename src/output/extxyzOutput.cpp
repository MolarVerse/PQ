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

#include "extxyzOutput.hpp"

#include <cstddef>
#include <initializer_list>
#include <ios>
#include <ostream>

#include "box.hpp"
#include "frame.hpp"

namespace out
{
    namespace
    {
        constexpr std::size_t EXTXYZ_VEL_WIDTH  = 20;
        constexpr int         LATTICE_PRECISION = 8;
        constexpr int         ENERGY_PRECISION  = 10;

        constexpr auto PROPERTIES =
            "species:S:1:pos:R:3:vel:R:3:forces:R:3:charge:R:1";

    }   // namespace

    /**
     * @brief write one frame
     *
     * @details the comment line holds the cell vectors, the step, the
     * potential energy and units=pq; atoms are written in molecule order
     *
     * @param frame
     * @param step
     */
    void ExtxyzOutput::write(const Frame &frame, const size_t step)
    {
        using textFormat::FIXED_WIDTH;
        using textFormat::NAME_WIDTH;
        using textFormat::PRECISION;

        const auto matrix = frame.box->getBoxMatrix();

        _buffer.clear();
        _buffer.left(frame.atoms.size(), 0).put('\n');

        _buffer.append("Lattice=\"");
        for (std::size_t vector = 0; vector < 3; ++vector)
        {
            for (std::size_t i = 0; i < 3; ++i)
            {
                if (vector + i > 0)
                {
                    _buffer.put(' ');
                }

                _buffer.fixed(matrix[i][vector], 0, LATTICE_PRECISION);
            }
        }

        _buffer.append("\" Properties=").append(PROPERTIES);
        _buffer.append(" pbc=\"T T T\" step=").left(step, 0);
        _buffer.append(" energy=")
            .fixed(frame.potentialEnergy, 0, ENERGY_PRECISION);
        _buffer.append(" units=pq\n");

        for (const auto &atom : frame.atoms)
        {
            const auto pos = frame.box->wrapPositionIntoBox(atom.position);

            _buffer.left(atom.name, NAME_WIDTH);

            for (const double value : {pos[0], pos[1], pos[2]})
            {
                _buffer.put(' ').fixed(value, FIXED_WIDTH, PRECISION);
            }

            const auto &vel   = atom.velocity;
            const auto &force = atom.force;

            for (const double value : {vel[0], vel[1], vel[2]})
            {
                _buffer.put(' ').scientific(value, EXTXYZ_VEL_WIDTH, PRECISION);
            }

            for (const double value : {force[0], force[1], force[2]})
            {
                _buffer.put(' ').fixed(value, FIXED_WIDTH, PRECISION);
            }

            _buffer.put(' ').fixed(atom.charge, FIXED_WIDTH, PRECISION);
            _buffer.put('\n');
        }

        const auto text = _buffer.view();
        _getFilePointer().write(
            text.data(),
            static_cast<std::streamsize>(text.size())
        );
        _getFilePointer() << std::flush;
    }

}   // namespace out
