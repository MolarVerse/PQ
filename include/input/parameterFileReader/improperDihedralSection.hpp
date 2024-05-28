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

#ifndef _IMPROPER_DIHEDRAL_SECTION_HPP_

#define _IMPROPER_DIHEDRAL_SECTION_HPP_

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
     * @class ImproperDihedralSection
     *
     * @brief reads improper dihedral section of parameter file
     *
     */
    class ImproperDihedralSection : public ParameterFileSection
    {
       public:
        [[nodiscard]] std::string keyword() override { return "impropers"; }

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

#endif   // _IMPROPER_DIHEDRAL_SECTION_HPP_