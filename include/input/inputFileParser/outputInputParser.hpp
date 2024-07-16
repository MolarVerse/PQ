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

#ifndef _OUTPUT_INPUT_PARSER_HPP_

#define _OUTPUT_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for pq::strings

namespace input
{
    /**
     * @class OutputInputParser inherits from InputFileParser
     *
     * @brief Parses the output commands in the input file
     *
     */
    class OutputInputParser : public InputFileParser
    {
       public:
        explicit OutputInputParser(pq::Engine &);

        void parseOutputFreq(const pq::strings &, const size_t);
        void parseFilePrefix(const pq::strings &, const size_t);

        void parseLogFilename(const pq::strings &, const size_t);
        void parseInfoFilename(const pq::strings &, const size_t);
        void parseEnergyFilename(const pq::strings &, const size_t);
        void parseInstantEnergyFilename(const pq::strings &, const size_t);
        void parseTrajectoryFilename(const pq::strings &, const size_t);
        void parseVelocityFilename(const pq::strings &, const size_t);
        void parseForceFilename(const pq::strings &, const size_t);
        void parseRestartFilename(const pq::strings &, const size_t);
        void parseChargeFilename(const pq::strings &, const size_t);
        void parseMomentumFilename(const pq::strings &, const size_t);

        void parseVirialFilename(const pq::strings &, const size_t);
        void parseStressFilename(const pq::strings &, const size_t);
        void parseBoxFilename(const pq::strings &, const size_t);
        void parseTimingsFilename(const pq::strings &, const size_t);
        void parseOptFilename(const pq::strings &, const size_t);

        void parseRPMDRestartFilename(const pq::strings &, const size_t);
        void parseRPMDTrajectoryFilename(const pq::strings &, const size_t);
        void parseRPMDVelocityFilename(const pq::strings &, const size_t);
        void parseRPMDForceFilename(const pq::strings &, const size_t);
        void parseRPMDChargeFilename(const pq::strings &, const size_t);
        void parseRPMDEnergyFilename(const pq::strings &, const size_t);
    };

}   // namespace input

#endif   // _OUTPUT_INPUT_PARSER_HPP_