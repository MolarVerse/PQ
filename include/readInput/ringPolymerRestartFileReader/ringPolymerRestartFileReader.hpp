#ifndef _RING_POLYMER_RESTART_FILE_READER_HPP_

#define _RING_POLYMER_RESTART_FILE_READER_HPP_

#include "ringPolymerEngine.hpp"

namespace readInput::ringPolymer
{
    void readRingPolymerRestartFile(engine::RingPolymerEngine &);

    /**
     * @class RingPolymerRestartFileReader
     *
     * @brief Reads a .rpmd.rst file sets the ring polymer beads in the engine
     *
     */
    class RingPolymerRestartFileReader
    {
      private:
        const std::string          _fileName;
        std::ifstream              _fp;
        engine::RingPolymerEngine &_engine;

      public:
        RingPolymerRestartFileReader(const std::string &fileName, engine::RingPolymerEngine &engine)
            : _fileName(fileName), _fp(fileName), _engine(engine){};

        void read();
    };
}   // namespace readInput::ringPolymer

#endif   // _RING_POLYMER_RESTART_FILE_READER_HPP_