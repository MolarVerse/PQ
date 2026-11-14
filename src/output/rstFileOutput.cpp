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

#include "rstFileOutput.hpp"

#include <cstddef>
#include <format>
#include <initializer_list>
#include <ostream>
#include <sstream>

#include "frame.hpp"

namespace out
{

    namespace
    {
        /**
         * @brief write Nose-Hoover thermostat chi/zeta info to the restart file
         *
         * @param frame
         * @param buffer
         */
        void writeNHChain(const Frame &frame, std::ostringstream &buffer)
        {
            const auto &chain = frame.noseHooverChain;

            for (std::size_t i = 0; i < chain.size(); ++i)
            {
                const auto &[chi, zeta] = chain[i];

                buffer << "chi "
                       << std::format(
                              "{:2d}\t{:10.5e}\t{:10.5e}",
                              i + 1,
                              chi,
                              zeta
                          )
                       << '\n';
            }
        }

        constexpr std::size_t NAME_WIDTH     = 5;
        constexpr std::size_t FIXED_WIDTH    = 15;
        constexpr std::size_t VELOCITY_WIDTH = 19;
        constexpr int         PRECISION      = 8;

    }   // namespace

    /**
     * @brief Write the restart file
     *
     * @param frame
     * @param step
     */
    void RstFileOutput::write(const Frame &frame, size_t step)
    {
        std::ostringstream buffer;

        buffer << "Step " << step << '\n';

        buffer << "Box   " << frame.box->get_boxDimensions() << "  "
               << frame.box->getBoxAngles() << '\n';

        writeNHChain(frame, buffer);

        _buffer.clear();
        _buffer.append(buffer.str());

        for (const auto &atom : frame.atoms)
        {
            const auto &pos   = atom.position;
            const auto &vel   = atom.velocity;
            const auto &force = atom.force;

            _buffer.left(atom.name, NAME_WIDTH).put('\t');
            _buffer.left(atom.indexInMolecule + 1, NAME_WIDTH).put('\t');
            _buffer.left(atom.moltype.get(), NAME_WIDTH).put('\t');

            for (const double value : {pos[0], pos[1], pos[2]})
                _buffer.fixed(value, FIXED_WIDTH, PRECISION).put('\t');

            for (const double value : {vel[0], vel[1], vel[2]})
                _buffer.scientific(value, VELOCITY_WIDTH, PRECISION).put('\t');

            _buffer.fixed(force[0], FIXED_WIDTH, PRECISION).put('\t');
            _buffer.fixed(force[1], FIXED_WIDTH, PRECISION).put('\t');
            _buffer.fixed(force[2], FIXED_WIDTH, PRECISION).put('\n');
        }

        _replaceFile(_buffer.view());
    }

}   // namespace out
