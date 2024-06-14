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

#ifndef _INPUT_FILE_PARSER_OUTPUT_HPP_

#define _INPUT_FILE_PARSER_OUTPUT_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser

namespace engine
{
    class Engine;   // Forward declaration
}

namespace input
{
    using strings = std::vector<std::string>;

    /**
     * @class InputFileParserOutput inherits from InputFileParser
     *
     * @brief Parses the output commands in the input file
     *
     */
    class InputFileParserOutput : public InputFileParser
    {
       public:
        explicit InputFileParserOutput(engine::Engine &);

        void parseOutputFreq(const strings &, const size_t);
        void parseFilePrefix(const strings &, const size_t);

        void parseLogFilename(const strings &, const size_t);
        void parseInfoFilename(const strings &, const size_t);
        void parseEnergyFilename(const strings &, const size_t);
        void parseInstantEnergyFilename(const strings &, const size_t);
        void parseTrajectoryFilename(const strings &, const size_t);
        void parseVelocityFilename(const strings &, const size_t);
        void parseForceFilename(const strings &, const size_t);
        void parseRestartFilename(const strings &, const size_t);
        void parseChargeFilename(const strings &, const size_t);
        void parseMomentumFilename(const strings &, const size_t);

        void parseVirialFilename(const strings &, const size_t);
        void parseStressFilename(const strings &, const size_t);
        void parseBoxFilename(const strings &, const size_t);

        void parseRPMDRestartFilename(const strings &, const size_t);
        void parseRPMDTrajectoryFilename(const strings &, const size_t);
        void parseRPMDVelocityFilename(const strings &, const size_t);
        void parseRPMDForceFilename(const strings &, const size_t);
        void parseRPMDChargeFilename(const strings &, const size_t);
        void parseRPMDEnergyFilename(const strings &, const size_t);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_OUTPUT_HPP_