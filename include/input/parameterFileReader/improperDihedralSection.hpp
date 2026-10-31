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

#ifndef INPUT__PARAMETER_FILE_READER__IMPROPER_DIHEDRAL_SECTION_HPP
#define INPUT__PARAMETER_FILE_READER__IMPROPER_DIHEDRAL_SECTION_HPP

#include <string>

#include "parameterFileSection.hpp"

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
        [[nodiscard]] std::string keyword() override;

        void processSection(
            std::vector<std::string> &lineElements,
            engine::Engine           &engine,
            size_t                    lineNumber
        ) override;
        void processHeader(
            std::vector<std::string> & /* lineElements */,
            engine::Engine & /* engine */,
            size_t /* lineNumber */
        ) override
        {
            // TODO(97gamjak): https://github.com/MolarVerse/PQ/issues/905
            // implement processHeader
        }
    };

}   // namespace input::parameterFile

#endif   // INPUT__PARAMETER_FILE_READER__IMPROPER_DIHEDRAL_SECTION_HPP
