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

#ifndef _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#define _INPUT_FILE_PARSER_RESET_KINETICS_HPP_

#include "inputFileParser.hpp"

#include <cstddef>   // for size_t
#include <string>
#include <vector>

namespace engine
{
    class Engine;   // Forward declaration
}

namespace input
{
    /**
     * @class InputFileParserResetKinetics inherits from InputFileParser
     *
     * @brief Parses the reset kinetics commands in the input file
     *
     */
    class InputFileParserResetKinetics : public InputFileParser
    {
      public:
        explicit InputFileParserResetKinetics(engine::Engine &);

        void parseNScale(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseFScale(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseNReset(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseFReset(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseNResetAngular(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseFResetAngular(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_RESET_KINETICS_HPP_