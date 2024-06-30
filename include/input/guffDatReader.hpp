/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _GUFF_DAT_READER_HPP_

#define _GUFF_DAT_READER_HPP_

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "defaults.hpp"   // for _GUFF_FILENAME_DEFAULT_
#include "typeAliases.hpp"

namespace input::guffdat
{
    void               readGuffDat(pq::Engine &);
    [[nodiscard]] bool isNeeded(pq::Engine &);

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
        std::string _fileName   = defaults::_GUFF_FILE_DEFAULT_;

        pq::stlVector4d     _guffCoulombCoeffs;
        pq::stlVector4dBool _isGuffPairSet;

        pq::Engine &_engine;

       public:
        explicit GuffDatReader(pq::Engine &engine);

        void setupGuffMaps();
        void parseLine(const pq::strings &lineCommands);
        void read();
        void postProcessSetup();
        void calculatePartialCharges();
        void checkPartialCharges();
        void checkNecessaryGuffPairs();
        void addNonCoulombPair(
            const size_t               molType1,
            const size_t               molType2,
            const size_t               atomType1,
            const size_t               atomType2,
            const std::vector<double> &coefficients,
            const double               rncCutOff
        );
        void addLennardJonesPair(
            const size_t               molType1,
            const size_t               molType2,
            const size_t               atomType1,
            const size_t               atomType2,
            const std::vector<double> &coefficients,
            const double               rncCutOff
        );
        void addBuckinghamPair(
            const size_t               molType1,
            const size_t               molType2,
            const size_t               atomType1,
            const size_t               atomType2,
            const std::vector<double> &coefficients,
            const double               rncCutOff
        );
        void addMorsePair(
            const size_t               molType1,
            const size_t               molType2,
            const size_t               atomType1,
            const size_t               atomType2,
            const std::vector<double> &coefficients,
            const double               rncCutOff
        );
        void addGuffPair(
            const size_t               molType1,
            const size_t               molType2,
            const size_t               atomType1,
            const size_t               atomType2,
            const std::vector<double> &coefficients,
            const double               rncCutOff
        );

        /********************
         * standard setters *
         ********************/

        void setFilename(const std::string_view &filename);
        void setGuffCoulombCoefficients(
            const size_t molType1,
            const size_t molType2,
            const size_t atomType1,
            const size_t atomType2,
            const double coefficient
        );
        void setIsGuffPairSet(
            const size_t molType1,
            const size_t molType2,
            const size_t atomType1,
            const size_t atomType2,
            const bool   isSet
        );

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] pq::stlVector4d     &getGuffCoulombCoefficients();
        [[nodiscard]] pq::stlVector4dBool &getIsGuffPairSet();
    };

}   // namespace input::guffdat

#endif   // _GUFF_DAT_READER_HPP_