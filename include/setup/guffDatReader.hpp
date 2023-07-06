#ifndef _GUFF_DAT_READER_H_

#define _GUFF_DAT_READER_H_

#include "engine.hpp"

#include <string>

namespace setup
{
    class GuffDatReader;
    void readGuffDat(engine::Engine &);
}   // namespace setup

/**
 * @class GuffDatReader
 *
 * @brief reads the guff.dat file
 *
 */
class setup::GuffDatReader
{
  private:
    int         _lineNumber = 1;
    std::string _filename   = "guff.dat";

    engine::Engine &_engine;

  public:
    explicit GuffDatReader(engine::Engine &engine) : _engine(engine) {}

    void setupGuffMaps();
    void parseLine(std::vector<std::string> &);
    void read();
    void setFilename(const std::string &filename) { _filename = filename; }
};

#endif   // _GUFF_DAT_READER_H_