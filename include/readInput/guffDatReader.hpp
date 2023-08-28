#ifndef _GUFF_DAT_READER_HPP_

#define _GUFF_DAT_READER_HPP_

#include "defaults.hpp"   // for _GUFF_FILENAME_DEFAULT_

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    void readGuffDat(engine::Engine &);

    using c_ul     = const size_t;
    using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

    /**
     * @class GuffDatReader
     *
     * @brief reads the guff.dat file
     *
     */
    class GuffDatReader
    {
      private:
        size_t      _lineNumber = 1;
        std::string _fileName   = defaults::_GUFF_FILENAME_DEFAULT_;   // gets overridden by the engine in the constructor

        vector4d _guffCoulombCoefficients;

        engine::Engine &_engine;

      public:
        explicit GuffDatReader(engine::Engine &engine);

        void setupGuffMaps();
        void parseLine(std::vector<std::string> &);
        void read();
        void postProcessSetup();

        void setFilename(const std::string_view &filename) { _fileName = filename; }
    };

}   // namespace readInput

#endif   // _GUFF_DAT_READER_HPP_