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

#ifndef _TOPOLOGY_READER_HPP_

#define _TOPOLOGY_READER_HPP_

#include "topologySection.hpp"

#include <fstream>   // for ifstream
#include <memory>
#include <string>
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;   // forward declaration
}

namespace readInput::topology
{
    void readTopologyFile(engine::Engine &);

    /**
     * @class TopologyReader
     *
     * @brief reads topology file and sets settings
     *
     */
    class TopologyReader
    {
      private:
        std::string     _fileName;
        std::ifstream   _fp;
        engine::Engine &_engine;

        std::vector<std::unique_ptr<TopologySection>> _topologySections;

      public:
        TopologyReader(const std::string &filename, engine::Engine &engine);

        void                           read();
        [[nodiscard]] bool             isNeeded() const;
        [[nodiscard]] TopologySection *determineSection(const std::vector<std::string> &lineElements);

        void setFilename(const std::string_view &filename) { _fileName = filename; }
    };

}   // namespace readInput::topology

#endif   // _TOPOLOGY_READER_HPP_