#ifndef _GUFF_DAT_READER_HPP_

#define _GUFF_DAT_READER_HPP_

#include "defaults.hpp"   // for _GUFF_FILENAME_DEFAULT_

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput::guffdat
{
    void readGuffDat(engine::Engine &);

    using c_ul         = const size_t;
    using vector4d     = std::vector<std::vector<std::vector<std::vector<double>>>>;
    using vector4dBool = std::vector<std::vector<std::vector<std::vector<bool>>>>;

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
        std::string _fileName   = defaults::_GUFF_FILENAME_DEFAULT_;

        vector4d     _guffCoulombCoefficients;
        vector4dBool _isGuffPairSet;

        engine::Engine &_engine;

      public:
        explicit GuffDatReader(engine::Engine &engine);

        void setupGuffMaps();
        void parseLine(const std::vector<std::string> &lineCommands);
        void read();
        void postProcessSetup();
        void checkPartialCharges();
        void addNonCoulombPair(const size_t               molType1,
                               const size_t               molType2,
                               const size_t               atomType1,
                               const size_t               atomType2,
                               const std::vector<double> &coefficients,
                               const double               rncCutOff);
        void addLennardJonesPair(const size_t               molType1,
                                 const size_t               molType2,
                                 const size_t               atomType1,
                                 const size_t               atomType2,
                                 const std::vector<double> &coefficients,
                                 const double               rncCutOff);
        void addBuckinghamPair(const size_t               molType1,
                               const size_t               molType2,
                               const size_t               atomType1,
                               const size_t               atomType2,
                               const std::vector<double> &coefficients,
                               const double               rncCutOff);
        void addMorsePair(const size_t               molType1,
                          const size_t               molType2,
                          const size_t               atomType1,
                          const size_t               atomType2,
                          const std::vector<double> &coefficients,
                          const double               rncCutOff);
        void addGuffPair(const size_t               molType1,
                         const size_t               molType2,
                         const size_t               atomType1,
                         const size_t               atomType2,
                         const std::vector<double> &coefficients,
                         const double               rncCutOff);

        void setFilename(const std::string_view &filename) { _fileName = filename; }

        [[nodiscard]] vector4d     &getGuffCoulombCoefficients() { return _guffCoulombCoefficients; }
        [[nodiscard]] vector4dBool &getIsGuffPairSet() { return _isGuffPairSet; }
    };

}   // namespace readInput::guffdat

#endif   // _GUFF_DAT_READER_HPP_