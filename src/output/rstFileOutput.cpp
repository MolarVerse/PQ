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
            const auto &chi  = frame.noseHooverChi;
            const auto &zeta = frame.noseHooverZeta;

            for (size_t i = 0; i < chi.size(); ++i)
            {
                buffer << "chi "
                       << std::format(
                              "{:2d}\t{:10.5e}\t{:10.5e}",
                              i + 1,
                              chi[i],
                              zeta[i]
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

        _getFilePointer().close();

        _openFile();

        buffer << "Step " << step << '\n';

        buffer << "Box   " << frame.boxDimensions << "  " << frame.boxAngles
               << '\n';

        if (!frame.noseHooverChi.empty())
            writeNHChain(frame, buffer);

        std::size_t atom = 0;

        for (const auto &[molType, nAtoms] : frame.molecules)
        {
            for (std::size_t i = 0; i < nAtoms; ++i, ++atom)
            {
                const auto x      = frame.positions[atom][0];
                const auto y      = frame.positions[atom][1];
                const auto z      = frame.positions[atom][2];
                const auto velX   = frame.velocities[atom][0];
                const auto velY   = frame.velocities[atom][1];
                const auto velZ   = frame.velocities[atom][2];
                const auto forceX = frame.forces[atom][0];
                const auto forceY = frame.forces[atom][1];
                const auto forceZ = frame.forces[atom][2];

                buffer << std::format("{:<5}\t", frame.names[atom]);
                buffer << std::format("{:<5}\t", i + 1);
                buffer << std::format("{:<5}\t", molType.get());

                buffer
                    << std::format("{:15.8f}\t{:15.8f}\t{:15.8f}\t", x, y, z);
                buffer << std::format(
                    "{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                    velX,
                    velY,
                    velZ
                );
                buffer << std::format(
                    "{:15.8f}\t{:15.8f}\t{:15.8f}",
                    forceX,
                    forceY,
                    forceZ
                );

                buffer << '\n' << std::flush;
            }
        }

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

}   // namespace out
