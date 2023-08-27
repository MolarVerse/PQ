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
        std::string     _filename;
        std::ifstream   _fp;
        engine::Engine &_engine;

        std::vector<std::unique_ptr<TopologySection>> _topologySections;

      public:
        TopologyReader(const std::string &filename, engine::Engine &engine);

        void                           read();
        [[nodiscard]] bool             isNeeded() const;
        [[nodiscard]] TopologySection *determineSection(const std::vector<std::string> &lineElements);

        void setFilename(const std::string_view &filename) { _filename = filename; }
    };

}   // namespace readInput::topology

#endif   // _TOPOLOGY_READER_HPP_