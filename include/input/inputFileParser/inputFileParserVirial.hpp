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

#ifndef _INPUT_FILE_PARSER_VIRIAL_HPP_

#define _INPUT_FILE_PARSER_VIRIAL_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace input
{
    /**
     * @class InputFileParserVirial inherits from InputFileParser
     *
     * @brief Parses the virial commands in the input file
     *
     */
    class InputFileParserVirial : public InputFileParser
    {
      public:
        explicit InputFileParserVirial(engine::Engine &);

        void parseVirial(const std::vector<std::string> &, const size_t);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_VIRIAL_HPP_