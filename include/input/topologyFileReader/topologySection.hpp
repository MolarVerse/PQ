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

#ifndef _TOPOLOGY_SECTION_HPP_

#define _TOPOLOGY_SECTION_HPP_

#include <iosfwd>   // for ifstream
#include <string>   // for string, allocator
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace input::topology
{
    /**
     * @class TopologySection
     *
     * @brief base class for reading topology file sections
     *
     */
    class TopologySection
    {
       protected:
        int            _lineNumber;
        std::ifstream *_fp;

       public:
        virtual ~TopologySection() = default;

        void process(std::vector<std::string> &lineElements, engine::Engine &);

        virtual std::string keyword() = 0;
        virtual void processSection(std::vector<std::string> &lineElements, engine::Engine &) = 0;
        virtual void endedNormally(const bool) const = 0;

        void setLineNumber(const int lineNumber) { _lineNumber = lineNumber; }
        void setFp(std::ifstream *fp) { _fp = fp; }

        [[nodiscard]] int getLineNumber() const { return _lineNumber; }
    };

}   // namespace input::topology

#endif   // _TOPOLOGY_SECTION_HPP_