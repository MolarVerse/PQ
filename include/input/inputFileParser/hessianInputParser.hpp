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

#ifndef _HESSIAN_INPUT_PARSER_HPP_

#define _HESSIAN_INPUT_PARSER_HPP_

#include <cstddef>

#include "inputFileParser.hpp"

namespace input
{
    class HessianInputParser : public InputFileParser
    {
       public:
        explicit HessianInputParser(engine::Engine &);

        void parseHessianFile(const std::vector<std::string> &, const size_t);
        void parseHessianInfoFile(
            const std::vector<std::string> &,
            const size_t
        );
        void parseDisplacement(const std::vector<std::string> &, const size_t);
        void parseOptimizeBeforeHessian(
            const std::vector<std::string> &,
            const size_t
        );
        void parseBuilder(const std::vector<std::string> &, const size_t);
    };

}   // namespace input

#endif   // _HESSIAN_INPUT_PARSER_HPP_
