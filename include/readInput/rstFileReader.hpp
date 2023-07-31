#ifndef _RST_FILE_READER_HPP_

#define _RST_FILE_READER_HPP_

#include "engine.hpp"
#include "rstFileSection.hpp"
#include "simulationBox.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace readInput
{
    class RstFileReader;
    void readRstFile(engine::Engine &);
}   // namespace readInput

/**
 * @class RstFileReader
 *
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 */
class readInput::RstFileReader
{
  private:
    const std::string _filename;
    std::ifstream     _fp;
    engine::Engine   &_engine;

    std::unique_ptr<readInput::RstFileSection>              _atomSection = std::make_unique<AtomSection>();
    std::vector<std::unique_ptr<readInput::RstFileSection>> _sections;

  public:
    RstFileReader(const std::string &, engine::Engine &);

    void                       read();
    readInput::RstFileSection *determineSection(std::vector<std::string> &);
};

#endif   // _RST_FILE_READER_HPP_