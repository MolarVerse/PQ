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
#include <ios>
#include <ostream>
#include <string_view>

#include "box.hpp"
#include "defaults.hpp"
#include "frame.hpp"
#include "hybridConfigurator.hpp"
#include "vector3d.hpp"

namespace out
{
    namespace
    {
        constexpr std::size_t NAME_WIDTH      = 5;
        constexpr std::size_t FIXED_WIDTH     = 15;
        constexpr std::size_t VELOCITY_WIDTH  = 20;
        constexpr int         PRECISION       = 8;

        /**
         * @brief append "name\tx\ty\tz\n" in fixed notation
         */
        void appendFixedLine(
            TextBuffer             &buffer,
            const std::string_view  name,
            const linalg::Vec3D    &vec
        )
        {
            buffer.left(name, NAME_WIDTH).put('\t');
            buffer.fixed(vec[0], FIXED_WIDTH, PRECISION).put('\t');
            buffer.fixed(vec[1], FIXED_WIDTH, PRECISION).put('\t');
            buffer.fixed(vec[2], FIXED_WIDTH, PRECISION).put('\n');
        }

    }   // namespace

    /**
     * @brief write the text buffer to the file and flush
     */
    void TrajectoryOutput::_flushBuffer()
    {
        const auto text = _buffer.view();
        _getFilePointer().write(
            text.data(),
            static_cast<std::streamsize>(text.size())
        );
        _getFilePointer() << std::flush;
    }

    /**
     * @brief Write the header of a trajectory files
     *
     * @param frame
     */
    void TrajectoryOutput::writeHeader(const Frame &frame)
    {
        _getFilePointer() << frame.atoms.size() << "  "
                          << frame.box->get_boxDimensions() << "  "
                          << frame.box->getBoxAngles() << '\n';
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
        writeHeader(frame);
        _writeComment(step, includeMetadata);

        _buffer.clear();

        for (const auto &atom : frame.atoms)
        {
            const auto pos = frame.box->wrapPositionIntoBox(atom.position);
            appendFixedLine(_buffer, atom.name, pos);
        }

        _flushBuffer();
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

        _buffer.clear();
        appendFixedLine(
            _buffer,
            std::string_view(&centerAtomName, 1),
            configurator.getInnerRegionCenter()
        );

        _flushBuffer();
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
        writeHeader(frame);
        _writeComment(step, includeMetadata);

        _buffer.clear();

        for (const auto &atom : frame.atoms)
        {
            const auto &vel = atom.velocity;

            _buffer.left(atom.name, NAME_WIDTH).put('\t');
            _buffer.scientific(vel[0], VELOCITY_WIDTH, PRECISION).put('\t');
            _buffer.scientific(vel[1], VELOCITY_WIDTH, PRECISION).put('\t');
            _buffer.scientific(vel[2], VELOCITY_WIDTH, PRECISION).put('\n');
        }

        _flushBuffer();
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
        writeHeader(frame);
        _writeForceComment(step, frame.totalForce(), includeMetadata);

        _buffer.clear();

        for (const auto &atom : frame.atoms)
            appendFixedLine(_buffer, atom.name, atom.force);

        _flushBuffer();
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
        writeHeader(frame);
        _writeComment(step, includeMetadata);

        _buffer.clear();

        for (const auto &atom : frame.atoms)
        {
            _buffer.left(atom.name, NAME_WIDTH).put('\t');
            _buffer.fixed(atom.charge, FIXED_WIDTH, PRECISION).put('\n');
        }

        _flushBuffer();
    }

}   // namespace out
