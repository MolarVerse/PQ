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

#include "parameterFileSection.hpp"
#include "typeAliases.hpp"

namespace input::parameterFile
{
    /**
     * @class BondSection
     *
     * @brief reads bond section of parameter file
     *
     */
    class BondSection : public ParameterFileSection
    {
       public:
        [[nodiscard]] std::string keyword() override;

        void processSection(pq::strings &, pq::Engine &) override;
        void processHeader(pq::strings &, pq::Engine &) override{};
        // TODO: implement processHeader
    };

}   // namespace input::parameterFile

#endif   // _BOND_SECTION_HPP_