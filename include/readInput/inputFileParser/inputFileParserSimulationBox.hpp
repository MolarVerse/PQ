/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

#define _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput
{
    /**
     * @class InputFileParserSimulationBox inherits from InputFileParser
     *
     * @brief Parses the simulation box commands in the input file
     *
     */
    class InputFileParserSimulationBox : public InputFileParser
    {
      public:
        explicit InputFileParserSimulationBox(engine::Engine &);

        void parseCoulombRadius(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseDensity(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseInitializeVelocities(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_SIMULATION_BOX_HPP_