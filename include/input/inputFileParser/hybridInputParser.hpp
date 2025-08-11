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

#ifndef _HYBRID_INPUT_PARSER_HPP_

#define _HYBRID_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for pq::strings

namespace input
{
    /**
     * @class HybridInputParser inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class HybridInputParser : public InputFileParser
    {
       public:
        explicit HybridInputParser(pq::Engine &);

        void parseInnerRegionCenter(const pq::strings &, const size_t);
        void parseForcedInnerList(const pq::strings &, const size_t);
        void parseForcedOuterList(const pq::strings &, const size_t);
        void parseUseQMCharges(const pq::strings &, const size_t);
        void parseCoreRadius(const pq::strings &, const size_t);
        void parseLayerRadius(const pq::strings &, const size_t);
        void parseSmoothingRegionThickness(const pq::strings &, const size_t);
    };

}   // namespace input

#endif   // _HYBRID_INPUT_PARSER_HPP_