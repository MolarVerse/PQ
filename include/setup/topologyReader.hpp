#ifndef _TOPOLOGY_READER_HPP_

#define _TOPOLOGY_READER_HPP_

#include "engine.hpp"
#include "topologySection.hpp"

#include <string>

namespace setup
{
    class TopologyReader;
    void readTopologyFile(const std::string &, engine::Engine &);

}   // namespace setup

/**
 * @class TopologyReader
 *
 * @brief reads topology file and sets settings
 *
 */
class setup::TopologyReader
{
  private:
    std::string     _filename;
    std::ifstream   _fp;
    engine::Engine &_engine;

    std::vector<setup::TopologySection *> _topologySections;

  public:
    TopologyReader(const std::string &filename, engine::Engine &engine);

    bool                    isNeeded() const;
    void                    read();
    setup::TopologySection *determineSection(const std::vector<std::string> &);

    void setFilename(const std::string &filename) { _filename = filename; }
};

/**
 * @brief constructs a TopologyReader and reads topology file
 *
 * @param filename
 * @param engine
 */
void setup::readTopologyFile(const std::string &filename, engine::Engine &engine)
{
    TopologyReader topologyReader(filename, engine);
    topologyReader.read();
}

#endif   // _TOPOLOGY_READER_HPP_