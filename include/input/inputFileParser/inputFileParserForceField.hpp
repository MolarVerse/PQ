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

#ifndef _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

#define _INPUT_FILE_PARSER_FORCE_FIELD_HPP_

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
     * @class InputFileParserForceField inherits from InputFileParser
     *
     * @brief Parses the force field commands in the input file
     *
     */
    class InputFileParserForceField : public InputFileParser
    {
       public:
        explicit InputFileParserForceField(engine::Engine &);

        void parseForceFieldType(const strings &, const size_t);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_FORCE_FIELD_HPP_