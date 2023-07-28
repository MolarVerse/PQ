#ifndef _TOPOLOGY_READER_HPP_

#define _TOPOLOGY_READER_HPP_

#include "engine.hpp"
#include "topologySection.hpp"

#include <string>

namespace readInput
{
    class TopologyReader;
    void readTopologyFile(engine::Engine &);

}   // namespace readInput

/**
 * @class TopologyReader
 *
 * @brief reads topology file and sets settings
 *
 */
class readInput::TopologyReader
{
  private:
    std::string     _filename;
    std::ifstream   _fp;
    engine::Engine &_engine;

    std::vector<readInput::TopologySection *> _topologySections;

  public:
    TopologyReader(const std::string &filename, engine::Engine &engine);

    bool                        isNeeded() const;
    void                        read();
    readInput::TopologySection *determineSection(const std::vector<std::string> &);

    void setFilename(const std::string &filename) { _filename = filename; }
};

#endif   // _TOPOLOGY_READER_HPP_