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

#ifndef _BOX_SECTION_HPP_

#define _BOX_SECTION_HPP_

#include "restartFileSection.hpp"

#include <string>   // for string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace input::restartFile
{
    /**
     * @class BoxSection
     *
     * @brief Reads the box section of a .rst file
     *
     */
    class BoxSection : public RestartFileSection
    {
      public:
        [[nodiscard]] std::string keyword() override { return "box"; }
        [[nodiscard]] bool        isHeader() override { return true; }
        void                      process(std::vector<std::string> &lineElements, engine::Engine &) override;
    };

}   // namespace input::restartFile

#endif   // _BOX_SECTION_HPP_