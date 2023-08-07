#ifndef _TOPOLOGY_READER_HPP_

#define _TOPOLOGY_READER_HPP_

#include "engine.hpp"
#include "topologySection.hpp"

#include <memory>
#include <string>

namespace readInput::topology
{
    class TopologyReader;
    void readTopologyFile(engine::Engine &);

}   // namespace readInput::topology

/**
 * @class TopologyReader
 *
 * @brief reads topology file and sets settings
 *
 */
class readInput::topology::TopologyReader
{
  private:
    std::string     _filename;
    std::ifstream   _fp;
    engine::Engine &_engine;

    std::vector<std::unique_ptr<readInput::topology::TopologySection>> _topologySections;

  public:
    TopologyReader(const std::string &filename, engine::Engine &engine);

    bool                                  isNeeded() const;
    void                                  read();
    readInput::topology::TopologySection *determineSection(const std::vector<std::string> &);

    void setFilename(const std::string_view &filename) { _filename = filename; }
};

#endif   // _TOPOLOGY_READER_HPP_