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

#ifndef _CONVERGENCE_INPUT_PARSER_HPP_

#define _CONVERGENCE_INPUT_PARSER_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

namespace input
{
    /**
     * @class ConvInputParser
     *
     * @brief Parses the input file for the optimizer
     *
     */
    class ConvInputParser : public InputFileParser
    {
       public:
        explicit ConvInputParser(engine::Engine &);

        void parseEnergyConvergenceStrategy(
            const std::vector<std::string> &,
            const size_t
        );

        void parseUseEnergyConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseUseForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseUseMaxForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseUseRMSForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseEnergyConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRelativeEnergyConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseAbsoluteEnergyConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseMaxForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );

        void parseRMSForceConvergence(
            const std::vector<std::string> &,
            const size_t
        );
    };

}   // namespace input

#endif   // _CONVERGENCE_INPUT_PARSER_HPP_
