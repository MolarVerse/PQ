#ifndef _INTRA_NON_BONDED_READER_HPP_

#define _INTRA_NON_BONDED_READER_HPP_

#include "engine.hpp"

namespace readInput
{
    class IntraNonBondedReader;
    void readIntraNonBondedFile(engine::Engine &);
}   // namespace readInput

/**
 * @class IntraNonBondedReader
 *
 * @brief reads the intra non bonded interactions from the intraNonBonded file
 */
class readInput::IntraNonBondedReader
{
  private:
    std::string   _filename;
    std::ifstream _fp;

    size_t _lineNumber;

    engine::Engine &_engine;

  public:
    IntraNonBondedReader(const std::string &filename, engine::Engine &engine)
        : _filename(filename), _fp(filename), _engine(engine)
    {
    }

    void read();
    void processMolecule(const size_t);
    bool isNeeded() const { return _engine.isIntraNonBondedActivated(); }
};

#endif   // _INTRA_NON_BONDED_READER_HPP_