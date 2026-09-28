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

#ifndef _RESET_KINETICS_INPUT_PARSER_HPP_

#define _RESET_KINETICS_INPUT_PARSER_HPP_

#include "inputFileParser.hpp"

namespace input
{
    /**
     * @brief ResetKineticsInputParser inherits from InputFileParser
     *
     * @details Parses the reset kinetics commands in the input file
     *
     */
    class ResetKineticsInputParser : public InputFileParser
    {
       public:
        ResetKineticsInputParser();

        void addNScaleKeyword();
        void addFScaleKeyword();
        void addNResetKeyword();
        void addFResetKeyword();
        void addNResetAngularKeyword();
        void addFResetAngularKeyword();
        void addFResetForcesKeyword();
    };

}   // namespace input

#endif   // _RESET_KINETICS_INPUT_PARSER_HPP_
