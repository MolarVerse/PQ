#ifndef _GUFF_DAT_READER_HPP_

#define _GUFF_DAT_READER_HPP_

#include "engine.hpp"

#include <string>

namespace readInput
{
    class GuffDatReader;
    void readGuffDat(engine::Engine &);
}   // namespace readInput

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

/**
 * @class GuffDatReader
 *
 * @brief reads the guff.dat file
 *
 */
class readInput::GuffDatReader
{
  private:
    size_t      _lineNumber = 1;
    std::string _filename   = defaults::_GUFF_FILENAME_DEFAULT_;   // gets overridden by the engine in the constructor

    vector4d _guffCoulombCoefficients;

    engine::Engine &_engine;

  public:
    explicit GuffDatReader(engine::Engine &engine);

    void setupGuffMaps();
    void parseLine(std::vector<std::string> &);
    void read();
    void postProcessSetup();

    void setFilename(const std::string_view &filename) { _filename = filename; }
};

#endif   // _GUFF_DAT_READER_HPP_