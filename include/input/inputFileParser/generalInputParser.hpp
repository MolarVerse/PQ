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

#ifndef _GENERAL_INPUT_PARSER_HPP_

#define _GENERAL_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for unique_ptr
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for pq::strings, pq::Engine

namespace input
{
    /**
     * @class GeneralInputParser inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class GeneralInputParser : public InputFileParser
    {
       public:
        explicit GeneralInputParser(pq::Engine &);

        void parseJobType(const pq::strings &, const size_t);
        void parseDimensionality(const pq::strings &, const size_t);
        void parseFloatingPointType(const pq::strings &, const size_t);
        void parseRandomSeed(const pq::strings &, const size_t);

        void parseJobTypeForEngine(
            const pq::strings &,
            const size_t,
            pq::UniqueEngine &
        );
    };

}   // namespace input

#endif   // _GENERAL_INPUT_PARSER_HPP_