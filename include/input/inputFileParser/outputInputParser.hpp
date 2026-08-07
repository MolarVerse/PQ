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

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for std::vector<std::string>

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

        void parseOverwriteOutput(
            const std::vector<std::string> &,
            const size_t
        );

        void parseIncludeOutputMetadata(
            const std::vector<std::string> &,
            const size_t
        );

        void parseOutputFreq(const std::vector<std::string> &, const size_t);

        void parseFilePrefix(const std::vector<std::string> &, const size_t);

        void parseLogFilename(const std::vector<std::string> &, const size_t);

        void parseRefFilename(const std::vector<std::string> &, const size_t);

        void parseInfoFilename(const std::vector<std::string> &, const size_t);

        void parseEnergyFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseInstantEnergyFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseTrajectoryFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseVelocityFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseForceFilename(const std::vector<std::string> &, const size_t);

        void parseRestartFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseChargeFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseMomentumFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseVirialFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseStressFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseBoxFilename(const std::vector<std::string> &, const size_t);

        void parseTimingsFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseOptFilename(const std::vector<std::string> &, const size_t);

        void parseRPMDRestartFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRPMDTrajectoryFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRPMDVelocityFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRPMDForceFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRPMDChargeFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRPMDEnergyFilename(
            const std::vector<std::string> &,
            const size_t
        );
    };

}   // namespace input

#endif   // _OUTPUT_INPUT_PARSER_HPP_
