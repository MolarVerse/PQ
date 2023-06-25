#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include "engine.hpp"
#include "rstFileSection.hpp"
#include "simulationBox.hpp"

#include <fstream>
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
    const std::string             _filename;
    std::ifstream                 _fp;
    engine::Engine               &_engine;
    std::vector<RstFileSection *> _sections;
    RstFileSection               *_atomSection = new AtomSection;

  public:
    RstFileReader(const std::string &, engine::Engine &);
    ~RstFileReader();

    void            read();
    RstFileSection *determineSection(std::vector<std::string> &);
};

#endif