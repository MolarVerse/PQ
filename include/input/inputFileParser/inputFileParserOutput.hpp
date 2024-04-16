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

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace input
{
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

        void parseOutputFreq(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseFilePrefix(const std::vector<std::string> &lineElements, const size_t lineNumber);

        void parseLogFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseInfoFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseEnergyFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);

        void parseVirialFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseStressFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);

        void parseRPMDRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRPMDTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRPMDVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRPMDForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRPMDChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseRPMDEnergyFilename(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_OUTPUT_HPP_