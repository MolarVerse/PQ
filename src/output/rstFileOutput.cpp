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
#include <ostream>
#include <sstream>

#include "frame.hpp"
#include "units.hpp"

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
                              chi.numerical_value_in(kcal_per_mol * s),
                              zeta.numerical_value_in(one)
                          )
                       << '\n';
            }
        }

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

        buffer << "Box   "
               << frame.box->get_boxDimensions().numerical_value_in(angstrom)
               << "  " << frame.box->getBoxAngles().numerical_value_in(deg)
               << '\n';

        writeNHChain(frame, buffer);

        for (const auto &atom : frame.atoms)
        {
            const auto &pos   = atom.position.numerical_value_in(angstrom);
            const auto &vel   = atom.velocity.numerical_value_in(angstrom / s);
            const auto &force = atom.force;

            buffer << std::format("{:<5}\t", atom.name);
            buffer << std::format("{:<5}\t", atom.indexInMolecule + 1);
            buffer << std::format("{:<5}\t", atom.moltype.get());

            buffer << std::format(
                "{:15.8f}\t{:15.8f}\t{:15.8f}\t",
                pos[0],
                pos[1],
                pos[2]
            );
            buffer << std::format(
                "{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                vel[0],
                vel[1],
                vel[2]
            );
            buffer << std::format(
                "{:15.8f}\t{:15.8f}\t{:15.8f}",
                force[0].numerical_value_in(kcal_per_mol / angstrom),
                force[1].numerical_value_in(kcal_per_mol / angstrom),
                force[2].numerical_value_in(kcal_per_mol / angstrom)
            );

            buffer << '\n' << std::flush;
        }

        _replaceFile(buffer.str());
    }

}   // namespace out
