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

#ifndef _FILES_INPUT_PARSER_HPP_

#define _FILES_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"

namespace input
{
    /**
     * @class FilesInputParser inherits from InputFileParser
     *
     * @brief Parses all input file commands related to input files
     *
     */
    class FilesInputParser : public InputFileParser
    {
       public:
        explicit FilesInputParser(pq::Engine &);

        void parseIntraNonBondedFile(const pq::strings &, const size_t);
        void parseTopologyFilename(const pq::strings &, const size_t);
        void parseParameterFilename(const pq::strings &, const size_t);
        void parseStartFilename(const pq::strings &, const size_t);
        void parseRingPolymerStartFilename(const pq::strings &, const size_t);
        void parseMoldescriptorFilename(const pq::strings &, const size_t);
        void parseGuffDatFilename(const pq::strings &, const size_t);
        void parseGuffPath(const pq::strings &, const size_t);
        void parseMShakeFilename(const pq::strings &, const size_t);
        void parseDFTBFilename(const pq::strings &, const size_t);
        void parseTMFilename(const pq::strings &, const size_t);
    };

}   // namespace input

#endif   // _FILES_INPUT_PARSER_HPP_