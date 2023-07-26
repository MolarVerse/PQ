#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include "engine.hpp"
#include "rstFileSection.hpp"
#include "simulationBox.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace setup
{
    class RstFileReader;
    void readRstFile(engine::Engine &);
}   // namespace setup

/**
 * @class RstFileReader
 *
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 */
class setup::RstFileReader
{
  private:
    const std::string _filename;
    std::ifstream     _fp;
    engine::Engine   &_engine;

    std::unique_ptr<setup::RstFileSection>              _atomSection = std::make_unique<AtomSection>();
    std::vector<std::unique_ptr<setup::RstFileSection>> _sections;

  public:
    RstFileReader(const std::string &, engine::Engine &);

    void                   read();
    setup::RstFileSection *determineSection(std::vector<std::string> &);
};

#endif