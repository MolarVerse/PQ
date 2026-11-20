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

#include <cmath>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <ostream>
#include <sstream>

#include "frame.hpp"
#include "units.hpp"

namespace out
{

    namespace
    {
        /**
         * @brief a value that the restart reader can read back
         *
         * @details the reader parses numbers with std::stod, which rejects
         * every subnormal number (below 2.2e-308) as out of range. Such a
         * value is zero for any physical purpose, so it is written as a zero
         * of the same sign; all other values are written exactly.
         */
        double readable(const double value)
        {
            if (std::fpclassify(value) == FP_SUBNORMAL)
                return std::copysign(0.0, value);

            return value;
        }

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
                              "{:2d}\t{}\t{}",
                              i + 1,
                              readable(chi),
                              readable(zeta)
                          )
                       << '\n';
            }
        }

        constexpr std::size_t RST_VELOCITY_WIDTH = 19;

    }   // namespace

    /**
     * @brief Write the restart file
     *
     * @param frame
     * @param step
     * @param finalTime the optional final simulation time to be written in the
     * restart file, optional because optimizer does not necessarily have a
     * well-defined simulation time
     */
    void RstFileOutput::write(
        const Frame              &frame,
        const size_t              step,
        const std::optional<Time> finalTime
    )
    {
        using textFormat::FIXED_WIDTH;
        using textFormat::NAME_WIDTH;

        std::ostringstream buffer;

        buffer << "Step " << step;
        if (finalTime.has_value())
        {
            buffer << " Simulation_Time " << finalTime->numerical_value_in(ps);
        }
        buffer << "\n";

        // std::format writes the shortest text that reads back as the same
        // double; the stream operator of Vector3D keeps 6 digits only, which
        // changes the box of a restarted run
        buffer << std::format(
            "Box   {}  {}\n",
            frame.box->get_boxDimensions(),
            frame.box->getBoxAngles()
        );

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
            {
                _buffer.exactFixed(readable(value), FIXED_WIDTH).put('\t');
            }

            for (const double value : {vel[0], vel[1], vel[2]})
            {
                _buffer.exactScientific(readable(value), RST_VELOCITY_WIDTH)
                    .put('\t');
            }

            _buffer.exactFixed(readable(force[0]), FIXED_WIDTH).put('\t');
            _buffer.exactFixed(readable(force[1]), FIXED_WIDTH).put('\t');
            _buffer.exactFixed(readable(force[2]), FIXED_WIDTH).put('\n');
        }

        _replaceFile(_buffer.view());
    }

}   // namespace out
