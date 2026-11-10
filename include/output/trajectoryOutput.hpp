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

#ifndef OUTPUT__TRAJECTORY_OUTPUT_HPP
#define OUTPUT__TRAJECTORY_OUTPUT_HPP

#include <cstddef>

#include "hybridConfigurator.hpp"
#include "output.hpp"

namespace configurator
{
    class HybridConfigurator;   // forward declaration
}   // namespace configurator

namespace out
{
    struct Frame;   // forward declaration

    /**
     * @brief TrajectoryOutput inherits from Output
     *
     * @details Output for xyz, vel, force, charges files
     *
     */
    class TrajectoryOutput : public Output
    {
       public:
        using Output::Output;

        void writeHeader(const Frame &frame);
        void writeXyz(const Frame &frame, size_t step, bool includeMetadata);
        void writeHybridCenterXyz(
            const configurator::HybridConfigurator &configurator,
            size_t                                  step,
            bool                                    includeMetadata
        );
        void writeVelocities(
            const Frame &frame,
            size_t       step,
            bool         includeMetadata
        );
        void writeForces(const Frame &frame, size_t step, bool includeMetadata);
        void writeCharges(
            const Frame &frame,
            size_t       step,
            bool         includeMetadata
        );
    };

}   // namespace out

#endif   // OUTPUT__TRAJECTORY_OUTPUT_HPP
