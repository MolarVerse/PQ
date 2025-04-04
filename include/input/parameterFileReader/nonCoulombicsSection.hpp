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

#ifndef _NON_COULOMBICS_SECTION_HPP_

#define _NON_COULOMBICS_SECTION_HPP_

#include <string>   // for allocator, string
#include <vector>   // for vector

#include "parameterFileSection.hpp"   // for ParameterFileSection
#include "typeAliases.hpp"

namespace input::parameterFile
{
    /**
     * @class NonCoulombicsSection
     *
     * @brief reads non-coulombics section of parameter file
     *
     */
    class NonCoulombicsSection : public ParameterFileSection
    {
       public:
        [[nodiscard]] std::string keyword() override;

        void processSection(pq::strings &, pq::Engine &) override;
        void processHeader(pq::strings &, pq::Engine &) override;
        void processLJ(pq::strings &, pq::Engine &) const;
        void processBuckingham(pq::strings &, pq::Engine &) const;
        void processMorse(pq::strings &, pq::Engine &) const;
    };

}   // namespace input::parameterFile

#endif   // _NON_COULOMBICS_SECTION_HPP_