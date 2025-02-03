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

#ifndef _RESET_KINETICS_INPUT_PARSER_HPP_

#define _RESET_KINETICS_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <string>
#include <vector>

#include "inputFileParser.hpp"
#include "typeAliases.hpp"

namespace input
{
    /**
     * @class ResetKineticsInputParser inherits from InputFileParser
     *
     * @brief Parses the reset kinetics commands in the input file
     *
     */
    class ResetKineticsInputParser : public InputFileParser
    {
       public:
        explicit ResetKineticsInputParser(pq::Engine &);

        void parseNScale(const pq::strings &, const size_t);
        void parseFScale(const pq::strings &, const size_t);
        void parseNReset(const pq::strings &, const size_t);
        void parseFReset(const pq::strings &, const size_t);
        void parseNResetAngular(const pq::strings &, const size_t);
        void parseFResetAngular(const pq::strings &, const size_t);
        void parseFResetForces(const pq::strings &, const size_t);
    };

}   // namespace input

#endif   // _RESET_KINETICS_INPUT_PARSER_HPP_