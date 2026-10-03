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

#ifndef _RESTART_FILE_SECTION_HPP_

#define _RESTART_FILE_SECTION_HPP_

#include <fstream>
#include <string>
#include <vector>

namespace engine
{
    class Engine;   // forward declaration
}   // namespace engine

namespace input::restartFile
{
    /**
     * @class RestartFileSection
     *
     * @brief Base class for all sections of a .rst file
     *
     */
    class RestartFileSection
    {
        // TODO: remove this public access
       protected:
        size_t         _lineNumber = 0;
        std::ifstream *_fp         = nullptr;

       public:
        virtual ~RestartFileSection() = default;

        virtual std::string keyword()  = 0;
        virtual bool        isHeader() = 0;
        virtual void        process(
                   std::vector<std::string> &lineElements,
                   engine::Engine &
               ) = 0;

        [[nodiscard]]
        size_t getLineNumber() const
        {
            return _lineNumber;
        }
        void setLineNumber(size_t lineNumber) { _lineNumber = lineNumber; }
        void setFilePointer(std::ifstream *file) { _fp = file; }
    };

}   // namespace input::restartFile

#endif   // _RESTART_FILE_SECTION_HPP_
