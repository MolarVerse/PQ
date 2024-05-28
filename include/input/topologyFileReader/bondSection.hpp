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

#ifndef _BOND_SECTION_HPP_

#define _BOND_SECTION_HPP_

#include <string>   // for allocator, string
#include <vector>   // for vector

#include "topologySection.hpp"   // for TopologySection

namespace engine
{
    class Engine;   // forward declaration
}

namespace input::topology
{
    /**
     * @class BondSection
     *
     * @brief reads bond section of topology file
     *
     */
    class BondSection : public TopologySection
    {
       public:
        [[nodiscard]] std::string keyword() override { return "bonds"; }
        void                      endedNormally(const bool) const override;

        void processSection(
            std::vector<std::string> &lineElements,
            engine::Engine           &engine
        ) override;
    };
}   // namespace input::topology

#endif   // _BOND_SECTION_HPP_