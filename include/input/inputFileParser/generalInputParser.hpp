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

#ifndef _GENERAL_INPUT_PARSER_HPP_

#define _GENERAL_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <memory>
#include <string>
#include <vector>

#include "inputFileParser.hpp"   // for InputFileParser

namespace engine
{
    class Engine;   // forward declaration
}   // namespace engine

namespace input
{
    /**
     * @class GeneralInputParser inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class GeneralInputParser : public InputFileParser
    {
       public:
        GeneralInputParser();

        void parseJobType(const std::vector<std::string> &, const size_t);

        void parseDimensionality(
            const std::vector<std::string> &,
            const size_t
        );

        void parseFloatingPointType(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRandomSeed(const std::vector<std::string> &, const size_t);

        static void parseJobTypeForEngine(
            const std::vector<std::string> &,
            const size_t,
            std::unique_ptr<engine::Engine> &
        );
    };

}   // namespace input

#endif   // _GENERAL_INPUT_PARSER_HPP_
