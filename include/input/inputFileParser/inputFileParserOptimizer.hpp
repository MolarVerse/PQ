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

#ifndef _INPUT_FILE_PARSER_OPTIMIZER_HPP_

#define _INPUT_FILE_PARSER_OPTIMIZER_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

namespace input
{
    using strings = std::vector<std::string>;

    /**
     * @class InputFileParserOptimizer
     *
     * @brief Parses the input file for the optimizer
     *
     */
    class InputFileParserOptimizer : public InputFileParser
    {
        explicit InputFileParserOptimizer(engine::Engine &);

        void parseOptimizer(const strings &, const size_t);
    };

}   // namespace input

#endif   // _INPUT_FILE_PARSER_OPTIMIZER_HPP_