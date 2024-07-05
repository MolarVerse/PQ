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

#ifndef _PARAMETER_FILE_SECTION_HPP_

#define _PARAMETER_FILE_SECTION_HPP_

#include <iosfwd>   // for ifstream
#include <string>   // for string, allocator
#include <vector>   // for vector

#include "typeAliases.hpp"
namespace input::parameterFile
{
    /**
     * @class ParameterFileSection
     *
     * @brief base class for reading parameter file sections
     *
     */
    class ParameterFileSection
    {
       protected:
        int            _lineNumber;
        std::ifstream *_fp;

       public:
        virtual ~ParameterFileSection() = default;

        virtual void process(pq::strings &lineElements, pq::Engine &);
        void         endedNormally(const bool);

        virtual std::string keyword() = 0;
        virtual void processSection(pq::strings &lineElements, pq::Engine &) = 0;
        virtual void processHeader(pq::strings &lineElements, pq::Engine &) = 0;

        void setLineNumber(const int lineNumber);
        void setFp(std::ifstream *fp);

        [[nodiscard]] int getLineNumber() const;
    };

}   // namespace input::parameterFile

#endif   // _PARAMETER_FILE_SECTION_HPP_