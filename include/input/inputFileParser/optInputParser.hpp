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

#ifndef _OPT_INPUT_PARSER_HPP_

#define _OPT_INPUT_PARSER_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

namespace input
{
    /**
     * @class OptInputParser
     *
     * @brief Parses the input file for the optimizer
     *
     */
    class OptInputParser : public InputFileParser
    {
       public:
        explicit OptInputParser(engine::Engine &);

        static void parseOptimizer(const std::vector<std::string> &, size_t);

        static void parseLearningRateStrategy(
            const std::vector<std::string> &,
            size_t
        );
        static void parseInitialLearningRate(
            const std::vector<std::string> &,
            size_t
        );
        static void parseLearningRateUpdateFreq(
            const std::vector<std::string> &,
            size_t
        );
        static void parseMinLearningRate(
            const std::vector<std::string> &,
            size_t
        );
        static void parseMaxLearningRate(
            const std::vector<std::string> &,
            size_t
        );

        static void parseLearningRateDecay(
            const std::vector<std::string> &,
            size_t
        );
    };

}   // namespace input

#endif   // _OPT_INPUT_PARSER_HPP_
