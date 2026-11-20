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

#ifndef INPUT__INPUT_FILE_PARSER__CONVERGENCE_INPUT_PARSER_HPP
#define INPUT__INPUT_FILE_PARSER__CONVERGENCE_INPUT_PARSER_HPP

#include <mstd/types.hpp>

#include "inputFileParser.hpp"

class ConvSettings;   // Forward declaration

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
        mstd::Ref<ConvSettings> _convSettings;

       public:
        explicit ConvInputParser(ConvSettings &convSettings);

        void addEnergyConvergenceStrategyKey();
        void addUseEnergyConvergenceKey();
        void addUseForceConvergenceKey();
        void addUseMaxForceConvergenceKey();
        void addUseRMSForceConvergenceKey();
        void addEnergyConvergenceKey();
        void addRelativeEnergyConvergenceKey();
        void addAbsoluteEnergyConvergenceKey();
        void addForceConvergenceKey();
        void addMaxForceConvergenceKey();
        void addRMSForceConvergenceKey();
    };

}   // namespace input

#endif   // INPUT__INPUT_FILE_PARSER__CONVERGENCE_INPUT_PARSER_HPP
