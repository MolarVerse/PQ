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

#include "trajectoryOutput.hpp"

#include <cstddef>
#include <format>
#include <ostream>
#include <sstream>

#include "box.hpp"
#include "defaults.hpp"
#include "frame.hpp"
#include "hybridConfigurator.hpp"

namespace out
{

    /**
     * @brief Write the header of a trajectory files
     *
     * @param frame
     */
    void TrajectoryOutput::writeHeader(const Frame &frame)
    {
        _getFilePointer() << frame.positions.size() << "  "
                          << frame.boxDimensions << "  " << frame.boxAngles
                          << '\n';
    }

    /**
     * @brief Write xyz file
     *
     * @param frame
     * @param step
     * @param includeMetadata whether to include metadata in the comment
     */
    void TrajectoryOutput::writeXyz(
        const Frame &frame,
        size_t       step,
        bool         includeMetadata
    )
    {
        std::ostringstream buffer;

        writeHeader(frame);
        _writeComment(step, includeMetadata);

        for (std::size_t i = 0; i < frame.positions.size(); ++i)
        {
            buffer << std::format("{:<5}\t", frame.names[i]);

            const auto &pos =
                frame.box->wrapPositionIntoBox(frame.positions[i]);

            buffer << std::format("{:15.8f}\t", pos[0]);
            buffer << std::format("{:15.8f}\t", pos[1]);
            buffer << std::format("{:15.8f}\n", pos[2]);
        }

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

    /**
     * @brief Write hybrid center xyz file
     *
     * @param configurator Hybrid configurator containing the inner region
     * center.
     * @param step
     * @param includeMetadata whether to include metadata in the comment
     *
     */
    void TrajectoryOutput::writeHybridCenterXyz(
        const configurator::HybridConfigurator &configurator,
        const size_t                            step,
        bool                                    includeMetadata
    )
    {
        // one dummy atom is needed to mark the inner region center
        constexpr size_t numberOfCenterAtoms = 1;
        constexpr char centerAtomName = defaults::INNER_REGION_CENTER_ATOM_NAME;

        // header line
        _getFilePointer() << numberOfCenterAtoms << '\n';
        _writeComment(step, includeMetadata);

        std::ostringstream buffer;
        buffer << std::format("{:<5}\t", centerAtomName);

        const auto &pos = configurator.getInnerRegionCenter();

        buffer << std::format("{:15.8f}\t", pos[0]);
        buffer << std::format("{:15.8f}\t", pos[1]);
        buffer << std::format("{:15.8f}\n", pos[2]);

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

    /**
     * @brief Write velocities file
     *
     * @param frame
     * @param step
     * @param includeMetadata whether to include metadata in the comment
     */
    void TrajectoryOutput::writeVelocities(
        const Frame &frame,
        size_t       step,
        bool         includeMetadata
    )
    {
        std::ostringstream buffer;

        writeHeader(frame);
        _writeComment(step, includeMetadata);

        for (std::size_t i = 0; i < frame.velocities.size(); ++i)
        {
            buffer << std::format("{:<5}\t", frame.names[i]);

            const auto &vel = frame.velocities[i];

            buffer << std::format("{:20.8e}\t", vel[0]);
            buffer << std::format("{:20.8e}\t", vel[1]);
            buffer << std::format("{:20.8e}\n", vel[2]);
        }

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

    /**
     * @brief Write forces file
     *
     * @param frame
     * @param step
     * @param includeMetadata whether to include metadata in the comment
     */
    void TrajectoryOutput::writeForces(
        const Frame &frame,
        size_t       step,
        bool         includeMetadata
    )
    {
        std::ostringstream buffer;

        writeHeader(frame);
        _writeForceComment(step, frame.totalForce, includeMetadata);

        for (std::size_t i = 0; i < frame.forces.size(); ++i)
        {
            buffer << std::format("{:<5}\t", frame.names[i]);

            const auto &force = frame.forces[i];

            buffer << std::format("{:15.8f}\t", force[0]);
            buffer << std::format("{:15.8f}\t", force[1]);
            buffer << std::format("{:15.8f}\n", force[2]);
        }

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

    /**
     * @brief Write charges file
     *
     * @param frame
     * @param step
     * @param includeMetadata whether to include metadata in the comment
     */
    void TrajectoryOutput::writeCharges(
        const Frame &frame,
        size_t       step,
        bool         includeMetadata
    )
    {
        std::ostringstream buffer;

        writeHeader(frame);
        _writeComment(step, includeMetadata);

        for (std::size_t i = 0; i < frame.charges.size(); ++i)
        {
            buffer << std::format("{:<5}\t", frame.names[i]);
            buffer << std::format("{:15.8f}\n", frame.charges[i]);
            buffer << std::flush;
        }

        // Write the buffer to the file
        _getFilePointer() << buffer.str();
        _getFilePointer() << std::flush;
    }

}   // namespace out
