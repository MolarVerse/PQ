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

#ifndef _J_COUPLING_SECTION_HPP_

#define _J_COUPLING_SECTION_HPP_

#include <string>   // for allocator, string
#include <vector>   // for vector

#include "parameterFileSection.hpp"   // for ParameterFileSection

namespace engine
{
    class Engine;   // forward declaration
}

namespace input::parameterFile
{
    /**
     * @class JCouplingSection
     *
     * @brief reads j-coupling section of parameter file
     *
     */
    class JCouplingSection : public ParameterFileSection
    {
       public:
        [[nodiscard]] std::string keyword() override { return "j-couplings"; }

        void processSection(
            std::vector<std::string> &lineElements,
            engine::Engine           &engine
        ) override;

        void processHeader(
            [[maybe_unused]] std::vector<std::string> &lineElements,
            [[maybe_unused]] engine::Engine           &engine
        ) override {};   // TODO: implement
    };

}   // namespace input::parameterFile

#endif   // _J_COUPLING_SECTION_HPP_