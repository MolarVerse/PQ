/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _RESTART_FILE_SECTION_HPP_

#define _RESTART_FILE_SECTION_HPP_

#include <fstream>   // for ifstream
#include <string>    // for string, allocator
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput::restartFile
{
    /**
     * @class RestartFileSection
     *
     * @brief Base class for all sections of a .rst file
     *
     */
    class RestartFileSection
    {
      public:
        virtual ~RestartFileSection() = default;

        int                 _lineNumber;
        std::ifstream      *_fp;
        virtual std::string keyword()                                                         = 0;
        virtual bool        isHeader()                                                        = 0;
        virtual void        process(std::vector<std::string> &lineElements, engine::Engine &) = 0;
    };

}   // namespace readInput::restartFile

#endif   // _RESTART_FILE_SECTION_HPP_