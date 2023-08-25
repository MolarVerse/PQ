#ifndef _INTRA_NON_BONDED_READER_HPP_

#define _INTRA_NON_BONDED_READER_HPP_

#include "engine.hpp"   // for Engine

#include <iosfwd>     // for ifstream
#include <stddef.h>   // for size_t
#include <string>     // for string

namespace readInput
{
    void readIntraNonBondedFile(engine::Engine &);

    /**
     * @class IntraNonBondedReader
     *
     * @brief reads the intra non bonded interactions from the intraNonBonded file
     */
    class IntraNonBondedReader
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

}   // namespace readInput

#endif   // _INTRA_NON_BONDED_READER_HPP_