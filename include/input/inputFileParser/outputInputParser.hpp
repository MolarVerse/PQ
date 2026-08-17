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
        explicit OutputInputParser(engine::Engine &);

        static void parseOverwriteOutput(
            const std::vector<std::string> &,
            size_t
        );

        static void parseIncludeOutputMetadata(
            const std::vector<std::string> &,
            size_t
        );

        static void parseOutputFreq(const std::vector<std::string> &, size_t);

        static void parseFilePrefix(const std::vector<std::string> &, size_t);

        static void parseLogFilename(const std::vector<std::string> &, size_t);

        static void parseRefFilename(const std::vector<std::string> &, size_t);

        static void parseInfoFilename(const std::vector<std::string> &, size_t);

        static void parseEnergyFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseInstantEnergyFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseTrajectoryFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseHybridCenterFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseVelocityFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseForceFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRestartFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseChargeFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseMomentumFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseVirialFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseStressFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseBoxFilename(const std::vector<std::string> &, size_t);

        static void parseTimingsFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseOptFilename(const std::vector<std::string> &, size_t);

        static void parseRPMDRestartFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRPMDTrajectoryFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRPMDVelocityFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRPMDForceFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRPMDChargeFilename(
            const std::vector<std::string> &,
            size_t
        );

        static void parseRPMDEnergyFilename(
            const std::vector<std::string> &,
            size_t
        );
    };

}   // namespace input

#endif   // _OUTPUT_INPUT_PARSER_HPP_
